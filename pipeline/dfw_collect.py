#!/usr/bin/env python3
"""
DFW Restaurant Data Collection Script
-------------------------------------
Scrapes Google Places for restaurants in the Dallas-Fort Worth metro, stores
results as JSON, and (optionally) downloads photos.

Key differences vs the exported notebook:
1. No Jupyter / IPython dependencies – it uses plain tqdm, logging, and prints.
2. All code is wrapped in functions so importing **does not** trigger API calls.
3. Command-line interface (argparse) to set restaurant count, output path, etc.
4. Robust logging instead of bare prints; can be redirected to a file.
5. Environment variable (or --api-key flag) for the Google Places API key.

Run `python dfw_collect.py -h` for options.
"""
import argparse
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from googlemaps import Client as GMapsClient  # type: ignore
from tqdm import tqdm

# Optional heavy deps – only imported if analysis is requested.
try:
    import matplotlib.pyplot as plt  # noqa: F401
    import geopandas as gpd  # noqa: F401
    from shapely.geometry import box  # noqa: F401
except ImportError:
    plt = gpd = box = None  # type: ignore

###############################################################################
# Configuration & Logging                                                     #
###############################################################################
DFW_BOUNDS = {
    "min_lat": 32.55,
    "max_lat": 33.05,
    "min_lng": -97.50,
    "max_lng": -96.55,
}

CUISINE_KEYWORDS = [
    "mexican restaurant",
    "italian restaurant",
    "chinese restaurant",
    "bbq restaurant",
    "steakhouse",
    "seafood restaurant",
    "thai restaurant",
    "indian restaurant",
    "sushi restaurant",
    "vietnamese restaurant",
    "korean restaurant",
    "tex-mex restaurant",
    "food truck",
    "diner",
    "cafe",
    "bakery",
    "breakfast",
    "brunch",
    "taqueria",
    "pizzeria",
]

PLACE_DETAILS_FIELDS = [
    "place_id",
    "name",
    "formatted_address",
    "geometry",
    "business_status",
    "vicinity",
    "formatted_phone_number",
    "website",
    "url",
    "rating",
    "user_ratings_total",
    "price_level",
    "current_opening_hours",
    "photo",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

###############################################################################
# Helpers                                                                     #
###############################################################################
class RateLimiter:
    """Simple leaky-bucket rate limiter for per-second and per-day quotas."""

    def __init__(self, calls_per_second: int = 3, calls_per_day: int | None = None):
        self.calls_per_second = calls_per_second
        self.calls_per_day = calls_per_day
        self._last_call = 0.0
        self._daily_calls = 0
        self._day_start = datetime.utcnow().date()

    def wait(self) -> bool:
        """Sleep if needed. Return False if daily limit exhausted."""
        now = datetime.utcnow()
        if now.date() != self._day_start:
            self._day_start = now.date()
            self._daily_calls = 0

        if self.calls_per_day and self._daily_calls >= self.calls_per_day:
            logger.warning("Daily API quota exhausted – aborting further calls.")
            return False

        # Ensure we respect per-second limit
        elapsed = time.time() - self._last_call
        min_interval = 1.0 / self.calls_per_second
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_call = time.time()
        self._daily_calls += 1
        return True


def backoff(api_fn, *args, rate_limiter: RateLimiter, retries: int = 5, **kwargs):
    """Call *api_fn* with exponential back-off and our rate limiter."""
    for attempt in range(retries):
        if not rate_limiter.wait():
            return None
        try:
            return api_fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if "OVER_QUERY_LIMIT" in str(exc):
                sleep = 2 ** attempt
                logger.warning("Rate limit hit – retrying in %s s", sleep)
                time.sleep(sleep)
                continue
            logger.error("API error: %s", exc)
            return None
    logger.error("Max retries exceeded for API call %s", api_fn.__name__)
    return None

###############################################################################
# Google Places helpers                                                       #
###############################################################################

def create_search_grid(bounds: Dict[str, float], tile_km: float = 1.0) -> List[Dict[str, Any]]:
    """Return a list of circular tiles (~tile_km radius) covering *bounds*."""
    lat_deg = tile_km / 111.0
    lng_deg = tile_km / (111.0 * math.cos(math.radians((bounds["min_lat"] + bounds["max_lat"]) / 2)))
    lat_tiles = math.ceil((bounds["max_lat"] - bounds["min_lat"]) / lat_deg)
    lng_tiles = math.ceil((bounds["max_lng"] - bounds["min_lng"]) / lng_deg)

    tiles: List[Dict[str, Any]] = []
    for i in range(lat_tiles):
        for j in range(lng_tiles):
            center_lat = bounds["min_lat"] + (i + 0.5) * lat_deg
            center_lng = bounds["min_lng"] + (j + 0.5) * lng_deg
            tiles.append({
                "center": (center_lat, center_lng),
                "radius": (tile_km * 1_000) / 2,  # metres
                "index": (i, j),
            })
    logger.info("Generated %d search tiles (%dx%d).", len(tiles), lat_tiles, lng_tiles)
    return tiles


def places_in_tile(gmaps: GMapsClient, tile: Dict[str, Any], rate: RateLimiter) -> List[Dict[str, Any]]:
    """Retrieve all pages of Nearby Search results for *tile*."""
    results: List[Dict[str, Any]] = []
    page_token: str | None = None
    for _page in range(3):
        if page_token:
            resp = backoff(gmaps.places, page_token=page_token, rate_limiter=rate)
        else:
            resp = backoff(
                gmaps.places_nearby,
                location=tile["center"],
                radius=tile["radius"],
                type="restaurant",
                rate_limiter=rate,
            )
        if not resp:
            break
        results.extend(resp.get("results", []))
        page_token = resp.get("next_page_token")
        if not page_token:
            break
        time.sleep(2)  # Google requirement between paginated requests
    return results


def collect_tiles(gmaps: GMapsClient, tiles: List[Dict[str, Any]], rate: RateLimiter,
                  limit: int) -> List[Dict[str, Any]]:
    """Collect restaurants from tiles until *limit* unique place_ids are reached."""
    collected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for tile in tqdm(tiles, desc="Geo tiles"):
        for res in places_in_tile(gmaps, tile, rate):
            pid = res.get("place_id")
            if pid and pid not in seen:
                seen.add(pid)
                res["source_tile"] = tile["index"]
                collected.append(res)
                if len(collected) >= limit:
                    return collected
    return collected


def search_keyword(gmaps: GMapsClient, keyword: str, location: Tuple[float, float],
                   rate: RateLimiter, radius: int = 25_000) -> List[Dict[str, Any]]:
    """Run a text search for *keyword* around *location*."""
    results: List[Dict[str, Any]] = []
    page_token: str | None = None
    for _ in range(2):
        if page_token:
            resp = backoff(gmaps.places, page_token=page_token, rate_limiter=rate)
        else:
            resp = backoff(
                gmaps.places,
                query=keyword,
                location=location,
                radius=radius,
                rate_limiter=rate,
            )
        if not resp:
            break
        results.extend(resp.get("results", []))
        page_token = resp.get("next_page_token")
        if not page_token:
            break
        time.sleep(2)
    for r in results:
        r["source_keyword"] = keyword
    return results


def collect_keywords(gmaps: GMapsClient, center: Tuple[float, float], rate: RateLimiter,
                     already_seen: set[str], needed: int) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for kw in tqdm(CUISINE_KEYWORDS, desc="Cuisine keywords"):
        for res in search_keyword(gmaps, kw, center, rate):
            pid = res.get("place_id")
            if pid and pid not in already_seen:
                already_seen.add(pid)
                collected.append(res)
                if len(collected) >= needed:
                    return collected
    return collected


def place_details(gmaps: GMapsClient, pid: str, rate: RateLimiter) -> Dict[str, Any] | None:
    resp = backoff(
        gmaps.place,
        place_id=pid,
        fields=PLACE_DETAILS_FIELDS,
        rate_limiter=rate,
    )
    return resp.get("result") if resp else None


def enrich_details(gmaps: GMapsClient, restaurants: List[Dict[str, Any]], rate: RateLimiter) -> List[Dict[str, Any]]:
    detailed: List[Dict[str, Any]] = []
    for r in tqdm(restaurants, desc="Details"):
        pid = r.get("place_id")
        if not pid:
            continue
        det = place_details(gmaps, pid, rate)
        if det:
            det.update({k: r[k] for k in ("source_tile", "source_keyword") if k in r})
            detailed.append(det)
    return detailed


def download_photo(reference: str, api_key: str, rate: RateLimiter, directory: Path) -> Path | None:
    params = {
        "maxwidth": 1600,
        "photoreference": reference,
        "key": api_key,
    }
    if not rate.wait():
        return None
    resp = requests.get("https://maps.googleapis.com/maps/api/place/photo", params=params, timeout=30)
    if resp.status_code != 200:
        logger.debug("Photo request failed: %s", resp.status_code)
        return None
    directory.mkdir(parents=True, exist_ok=True)
    filename = directory / f"{reference}.jpg"
    filename.write_bytes(resp.content)
    return filename

###############################################################################
# Pipeline                                                                    #
###############################################################################

def collect_dfw_restaurants(gmaps: GMapsClient, api_key: str, target: int, *,
                            photo_dir: Path | None = None) -> List[Dict[str, Any]]:
    rate = RateLimiter(calls_per_second=3)

    # 1. Geo-tile search
    tiles = create_search_grid(DFW_BOUNDS, tile_km=1.0)
    restaurants = collect_tiles(gmaps, tiles, rate, target)

    # 2. Keyword backfill if needed
    if len(restaurants) < target:
        center = ((DFW_BOUNDS["min_lat"] + DFW_BOUNDS["max_lat"]) / 2,
                  (DFW_BOUNDS["min_lng"] + DFW_BOUNDS["max_lng"]) / 2)
        remaining = target - len(restaurants)
        existing_pids = {r["place_id"] for r in restaurants if r.get("place_id")}
        extra = collect_keywords(gmaps, center, rate, existing_pids, remaining)
        restaurants.extend(extra)

    logger.info("Collected %d unique places.", len(restaurants))

    # 3. Details enrichment
    detailed = enrich_details(gmaps, restaurants, rate)

    # 4. Optional photos
    if photo_dir is not None:
        for det in tqdm(detailed, desc="Photos"):
            photos = det.get("photos", [])[:3]
            saved: List[str] = []
            for p in photos:
                ref = p.get("photo_reference")
                if not ref:
                    continue
                path = download_photo(ref, api_key, rate, photo_dir)
                if path:
                    saved.append(str(path))
            det["downloaded_photos"] = saved
    return detailed

###############################################################################
# CLI                                                                         #
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DFW restaurant data via Google Places API")
    parser.add_argument("--count", type=int, default=1_000, help="Number of restaurants to collect (default 1000)")
    parser.add_argument("--output", type=Path, default=Path("dfw_restaurants.json"), help="Where to save the JSON dataset")
    parser.add_argument("--photos", action="store_true", help="Download photos as well (quota heavy!)")
    parser.add_argument("--photo-dir", type=Path, default=Path("restaurant_photos"), help="Directory to store photos")
    parser.add_argument("--api-key", type=str, help="Google Places API key (overrides .env)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = args.api_key or os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        logger.error("GOOGLE_PLACES_API_KEY not provided – set it in .env or pass --api-key")
        raise SystemExit(1)

    gmaps = GMapsClient(key=api_key)

    try:
        dataset = collect_dfw_restaurants(
            gmaps,
            api_key,
            target=args.count,
            photo_dir=args.photo_dir if args.photos else None,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted – saving partial data.")
        dataset = []

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)
    logger.info("Dataset written to %s (%d restaurants).", args.output, len(dataset))


if __name__ == "__main__":
    main() 
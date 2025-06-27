# EatFW

Your AI guide for Fort Worth's best food.

## Technical Decisions

### 1. Speech-to-Text: Web Speech API

- **Decision**: For capturing and transcribing user voice input, we are using the browser-native [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API).
- **Reasoning**:
  - **Simplicity**: It's built directly into modern browsers, requiring no external dependencies, SDKs, or API keys for basic implementation. This allows for rapid prototyping and development.
  - **Real-time Feedback**: The API provides `interimResults`, which allows us to show the user a live transcription of their speech, improving the user experience.
  - **Cost-Effective**: As a browser-native API, it is free to use.
- **Trade-offs**:
  - **Browser Compatibility**: Support is not universal across all browsers (excellent in Chrome/Edge, less so in others). This is an acceptable trade-off for the initial version of this application.
  - **Accuracy**: Recognition accuracy depends on the browser's and underlying OS's implementation, which may be less accurate than specialized third-party services. If accuracy becomes a major issue, we can migrate to a cloud-based solution like Google Speech-to-Text or AssemblyAI in the future.

# 👽🔬 AI Research Buddy 🔭🧪

An interactive web application built with Streamlit designed to assist STEM researchers by leveraging AI for information discovery, content analysis, and graph generation.

## Project Description

An interactive Streamlit web app utilizing specialized AI agents (Gemini/Serper) for research support. Offers information extraction, content analysis, direct Q&A, data visualization, and a quirky chatbot.

## Abstract

Addresses research bottlenecks by providing scholars an AI assistant. Integrates web search (Serper) and large language models (Google Gemini) to accelerate information discovery, content analysis, and data visualization in STEM fields.

## Features

*   **📄 Extract Information:** Search the web and AI knowledge base for specific topics based on field, topic name, and a detailed prompt. Includes relevant images from web search.
*   **📝 Content Analysis & Summarization:** Upload files (`.txt`, `.pdf`, `.md`, `.docx`, `.csv`) or paste text for AI-powered summarization and analysis, with optional specific instructions.
*   **❓ Direct Query:** Ask direct questions to the AI, optionally using web search results for context and including relevant images. Domain focus is automatically detected.
*   **📊 Graph Generator:** Upload CSV data, choose columns and plot type (Line, Bar, Scatter), or let the AI suggest the best plot ("Default"). Generates the graph and provides an AI summary of key takeaways.
*   **🤖 Quantum Quips Corner 😂:** A conversational chatbot ("Buddy") with a witty, sometimes dark/sarcastic personality for lighter interaction and occasional AI-generated jokes.

## Technologies Used

*   **Frontend:** Streamlit
*   **Core AI:** Google Gemini API (via `google-generativeai` library)
*   **Web/Image Search:** Serper API (via `requests` library)
*   **Plotting:** Matplotlib
*   **Data Handling:** Pandas
*   **File Parsing:** PyMuPDF (`fitz`), python-docx
*   **Language:** Python 3

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd AI_Research_Buddy
    ```
    *(Replace `<your-repository-url>` with the actual URL after creating it on GitHub)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    *   Create a directory named `.streamlit` inside your project folder if it doesn't exist.
    *   Inside `.streamlit`, create a file named `secrets.toml`.
    *   Add your API keys to `secrets.toml` in the following format:
        ```toml
        GOOGLE_API_KEY = "your_google_api_key_here"
        SERPER_API_KEY = "your_serper_api_key_here"
        ```
    *   Replace the placeholder values with your actual keys.
        *   Get a Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
        *   Get a Serper API key from [Serper.dev](https://serper.dev/).

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```
    The app should open in your default web browser.

## Configuration

The application requires API keys stored in `.streamlit/secrets.toml`:

*   `GOOGLE_API_KEY`: For accessing the Google Gemini models.
*   `SERPER_API_KEY`: For accessing Serper's web and image search results.

Ensure these keys are correctly placed in the `secrets.toml` file before running the app.

## Usage

1.  Launch the app using `streamlit run app.py`.
2.  Use the sidebar buttons to select the desired task:
    *   **Extract Information:** Fill in the field, topic, prompt, and word limit.
    *   **Content Analysis & Summarization:** Choose file upload or paste text, upload/paste content, set optional limits/instructions.
    *   **Direct Query:** Enter your question, choose whether to use web search.
    *   **Graph Generator:** Upload a CSV, select plot type (or "Default"), specify columns (if not default) and optional guidance, click generate.
    *   **Quantum Quips Corner:** Interact with the chatbot or clear the history.
3.  Submit the forms or enter chat messages.
4.  View the results (text, images, sources, plot, summary) in the main panel.
5.  Use the "Clear Output/Plot/Chat" button within each section to reset its specific output area.

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue on the GitHub repository to discuss changes or report bugs.

<!-- Optional: Add Screenshots Here -->
<!--
## Screenshots
(Add screenshots of your app here if desired)
![App Screenshot 1](path/to/screenshot1.png)
![App Screenshot 2](path/to/screenshot2.png)
-->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
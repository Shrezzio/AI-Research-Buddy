# app.py
"""
Main Streamlit application file for the AI Research Buddy.

Handles the user interface (UI) elements, manages application state using
session state, orchestrates the workflow based on user selections, and
calls utility functions from `utils.py` for backend operations like API calls,
parsing, and plotting.
"""

import streamlit as st
import pandas as pd
import json     # For serializing chat history for API context
import random   # For adding randomness (e.g., chatbot behavior, joke styles)
from utils import (
    get_api_keys,           # Function to retrieve API keys
    call_gemini_api,        # Function to interact with Gemini API
    run_serper_search,      # Function for web text search
    search_images_serper,   # Function for single image search
    parse_uploaded_file,    # Master file parsing function
    classify_query_domain,  # Function to predict query domain
    format_final_prompt,    # Function to construct LLM prompts
    generate_plot,          # Function to create plots
    parse_plot_suggestion,  # Function to parse plot suggestions
    VALID_DOMAINS,          # Constant list of domains
    # PLOT_TYPES - No longer directly used here, UI_PLOT_TYPES is used
    BASE_PLOT_TYPES, # Base types for validation/generation
    UI_PLOT_TYPES    # Types shown in the UI dropdown
)

# --- Page Configuration (must be the first Streamlit command) ---
st.set_page_config(page_title="AI Research Buddy", layout="wide")

# --- Application Title ---
# Includes relevant emojis for visual appeal
st.title("👽🔬 AI Research Buddy 🔭🧪")

# --- API Key Retrieval and Validation ---
# Fetches keys early to determine feature availability
google_api_key, serper_api_key = get_api_keys()

# --- Session State Initialization ---
# Initialize all necessary session state variables if they don't exist.
# This prevents errors on the first run or after a hard refresh and preserves user work.

# State for general research tasks (text output, optional image, sources)
if 'response_text' not in st.session_state: st.session_state.response_text = None
if 'image_url' not in st.session_state: st.session_state.image_url = None
if 'sources' not in st.session_state: st.session_state.sources = []

# State for the Graph Generator task
if 'plot_image' not in st.session_state: st.session_state.plot_image = None # Generated plot image (bytes)
if 'graph_summary' not in st.session_state: st.session_state.graph_summary = None # AI summary of the graph
if 'dataframe' not in st.session_state: st.session_state.dataframe = None # Loaded pandas DataFrame
if 'columns' not in st.session_state: st.session_state.columns = []       # List of column names from the DataFrame
if 'current_file_id' not in st.session_state: st.session_state.current_file_id = None # Identifier for the loaded CSV

# State for the Chatbot feature
if 'chat_history' not in st.session_state: st.session_state.chat_history = [] # List of chat message dictionaries

# State for controlling which view/task is displayed
if 'current_view' not in st.session_state: st.session_state.current_view = "Extract Information" # Default view on load
if 'last_view' not in st.session_state: st.session_state.last_view = None # Tracks previous view

# State for storing graph parameters temporarily for consistent plot captioning
if 'plot_x' not in st.session_state: st.session_state.plot_x = None
if 'plot_y' not in st.session_state: st.session_state.plot_y = None
if 'plot_pt' not in st.session_state: st.session_state.plot_pt = None

# --- App Description ---
st.markdown("""
Welcome! This app helps you access and analyze research information using AI (**powered by Google Gemini**).
Choose a main operation from the sidebar.
""")

# --- State Reset Function ---
def reset_output_state_for_view(view_to_clear):
    """
    Clears the output-related session state variables for a specific view.
    Called when a 'Clear' button is pressed or before generating new output within a view.
    """
    # Clear state variables based on the view being cleared
    if view_to_clear == "Extract Information":
        st.session_state.response_text = None; st.session_state.image_url = None; st.session_state.sources = []
    elif view_to_clear == "Content Analysis & Summarization":
        st.session_state.response_text = None
    elif view_to_clear == "Direct Query":
        st.session_state.response_text = None; st.session_state.image_url = None; st.session_state.sources = []
    elif view_to_clear == "Graph Generator":
        st.session_state.plot_image = None; st.session_state.graph_summary = None
        st.session_state.plot_x = None; st.session_state.plot_y = None; st.session_state.plot_pt = None
        # Note: Loaded dataframe/columns are preserved unless explicitly cleared elsewhere (e.g., file change)
    elif view_to_clear == "Quantum Quips Corner":
        st.session_state.chat_history = [] # Only clears chat history


# --- Sidebar Navigation ---
st.sidebar.header("Select Task")

def set_view(view_name):
    """Callback function to update the current view in session state."""
    st.session_state.last_view = st.session_state.current_view # Store previous view
    st.session_state.current_view = view_name

# Buttons for selecting the main application view/task
if st.sidebar.button("📄 Extract Information", key="btn_extract", use_container_width=True): set_view("Extract Information"); st.rerun()
if st.sidebar.button("📝 Content Analysis & Summarization", key="btn_analyze", use_container_width=True): set_view("Content Analysis & Summarization"); st.rerun()
if st.sidebar.button("❓ Direct Query", key="btn_query", use_container_width=True): set_view("Direct Query"); st.rerun()
if st.sidebar.button("📊 Graph Generator", key="btn_graph", use_container_width=True): set_view("Graph Generator"); st.rerun()
if st.sidebar.button("🤖 Quantum Quips Corner 😂", key="btn_chat", use_container_width=True): set_view("Quantum Quips Corner"); st.rerun()

# Sidebar informational text about the app's backend
st.sidebar.divider()
st.sidebar.info("Powered by Google Gemini & Serper APIs.\nDeveloped for research assistance.")

# --- Main Application Panel ---

# Get the currently selected view from session state
current_view = st.session_state.current_view

# --- Render UI and Logic based on the Current View ---
# The main panel's content dynamically changes based on the `current_view` state.

if current_view == "Extract Information":
    st.subheader("Extract Information from Web & AI Knowledge")
    submitted_extract = False # Track if form was submitted this run
    with st.form("extract_form"):
        # Input widgets for information extraction parameters
        field_of_interest=st.selectbox("Field:",VALID_DOMAINS,index=VALID_DOMAINS.index("General"), key="extract_field")
        topic_name=st.text_input("Topic Name (context):",placeholder="e.g., Quantum Entanglement", key="extract_topic")
        user_prompt=st.text_area("Specific Request/Prompt (searches):",placeholder="e.g., Explain qubit superposition", key="extract_prompt")
        word_limit=st.number_input("Approx Output Word Limit:",min_value=0,value=500, key="extract_limit")
        submitted_extract = st.form_submit_button("Extract & Synthesize")

        if submitted_extract:
            reset_output_state_for_view(current_view) # Clear previous outputs for this view
            if not google_api_key or not serper_api_key: st.error("Google & Serper API Keys required.")
            elif not topic_name or not user_prompt: st.warning("Topic & Request required.")
            else:
                search_query_text=f"{topic_name} {user_prompt} {field_of_interest}"
                image_search_query=f"{topic_name} {user_prompt}"
                with st.spinner(f"Processing request..."): # User feedback during processing
                    st.session_state.image_url=search_images_serper(image_search_query, serper_api_key)
                    search_context, sources = run_serper_search(search_query_text, serper_api_key)
                    st.session_state.sources = sources # Store results in state
                    final_prompt=format_final_prompt(user_prompt=user_prompt, specialist_area=field_of_interest, search_context=search_context, task_type="extract", word_limit=word_limit if word_limit > 0 else None)
                    st.session_state.response_text=call_gemini_api(final_prompt, google_api_key) # Store results in state

    # Display Area for this view
    st.divider(); st.subheader("Output")
    if st.button("Clear Output", key="clear_extract_btn"): reset_output_state_for_view(current_view); st.rerun() # Manual clear button

    output_exists = st.session_state.response_text or st.session_state.image_url or st.session_state.sources
    if output_exists:
        if st.session_state.image_url:
            try: st.image(st.session_state.image_url, caption="Relevant Image", use_container_width=True); st.divider()
            except Exception as e: st.error(f"Image display error: {e}") # Graceful image error handling
        if st.session_state.response_text: st.markdown(st.session_state.response_text)
        if st.session_state.sources:
            st.subheader("Relevant Sources")
            for source in st.session_state.sources: st.markdown(f"- [{source.get('title', 'Source Link')}]({source.get('link', '#')})")
    elif submitted_extract and (google_api_key and serper_api_key and topic_name and user_prompt):
         st.info("Processing complete, but no specific output generated (APIs might have returned empty or failed).")


elif current_view == "Content Analysis & Summarization":
    st.subheader("Content Analysis & Summarization")
    source_type=st.radio("Input Source:",["Upload File","Paste Text"],key="analysis_source")
    uploaded_file=None; pasted_text=""
    if source_type == "Upload File": uploaded_file=st.file_uploader("Upload File:",type=["txt","pdf","md","docx","csv"], key="analyze_uploader")
    else: pasted_text=st.text_area("Paste Text:",height=250, key="analyze_paste")

    submitted_analysis = False # Track form submission
    with st.form("analysis_form"):
        awl=st.number_input("Approx Summary Word Limit:",min_value=0,value=300, key="analyze_limit")
        ai=st.text_area("Additional Instructions (Optional):",placeholder="e.g., Focus on methodology section", key="analyze_instr")
        submitted_analysis = st.form_submit_button("Analyze Content")

        if submitted_analysis:
             reset_output_state_for_view(current_view) # Clear previous output for this view
             if not google_api_key: st.error("Google API Key required for analysis.")
             else:
                content_to_analyze = None
                if source_type=="Upload File":
                    if uploaded_file:
                        with st.spinner(f"Parsing {uploaded_file.name}..."): content_to_analyze=parse_uploaded_file(uploaded_file)
                    else: st.warning("Please upload a file.")
                else: # Pasted Text
                    if pasted_text: content_to_analyze=pasted_text
                    else: st.warning("Please paste some text.")
                if content_to_analyze: # If content successfully obtained
                     with st.spinner("Analyzing content..."):
                         final_prompt=format_final_prompt(user_prompt="", specialist_area="General", content_to_analyze=content_to_analyze, task_type="analyze", word_limit=awl if awl > 0 else None, additional_instructions=ai)
                         st.session_state.response_text=call_gemini_api(final_prompt,google_api_key)

    # Display Area for this view
    st.divider(); st.subheader("Output")
    if st.button("Clear Output", key="clear_analyze_btn"): reset_output_state_for_view(current_view); st.rerun() # Manual clear button

    if st.session_state.response_text:
        st.markdown(st.session_state.response_text)
    elif submitted_analysis and google_api_key and ((source_type=="Upload File" and uploaded_file) or (source_type=="Paste Text" and pasted_text)) and content_to_analyze:
        st.info("Processing complete, but no specific output generated.")


elif current_view == "Direct Query":
    st.subheader("Ask a Direct Question")
    submitted_query = False # Track form submission
    with st.form("query_form"):
        uq=st.text_area("Question (searches):",height=150,placeholder="e.g., Ethical considerations of AI?", key="query_text")
        uws=st.checkbox("Web search context?",value=True, key="query_web_search")
        submitted_query = st.form_submit_button("Get Answer")

        if submitted_query:
             reset_output_state_for_view(current_view) # Clear previous output for this view
             if not google_api_key: st.error("Google API Key required.")
             elif not uq: st.warning("Please enter your question.")
             else:
                 specialist_area="General"
                 if google_api_key:
                     with st.spinner("Classifying domain..."): specialist_area=classify_query_domain(uq,google_api_key); st.info(f"Detected Domain Focus: {specialist_area}")
                 search_context = None
                 with st.spinner(f"Processing request..."):
                    if serper_api_key:
                        image_search_query=uq; text_search_query=f"{uq} {specialist_area}" if uws else None
                        st.session_state.image_url=search_images_serper(image_search_query, serper_api_key)
                        if uws and text_search_query: search_context, sources = run_serper_search(text_search_query, serper_api_key); st.session_state.sources=sources
                        else: st.session_state.sources=[]
                    else: st.session_state.sources=[]; st.session_state.image_url=None
                    if google_api_key:
                        final_prompt=format_final_prompt(user_prompt=uq,specialist_area=specialist_area,search_context=search_context,task_type="query")
                        st.session_state.response_text=call_gemini_api(final_prompt,google_api_key)

    # Display Area for this view
    st.divider(); st.subheader("Output")
    if st.button("Clear Output", key="clear_query_btn"): reset_output_state_for_view(current_view); st.rerun() # Manual clear button

    output_exists = st.session_state.response_text or st.session_state.image_url or st.session_state.sources
    if output_exists:
        if st.session_state.image_url:
            try: st.image(st.session_state.image_url, caption="Relevant Image", use_container_width=True); st.divider()
            except Exception as e: st.error(f"Img display error: {e}")
        if st.session_state.response_text: st.markdown(st.session_state.response_text)
        if st.session_state.sources:
            st.subheader("Relevant Sources")
            for source in st.session_state.sources: st.markdown(f"- [{source.get('title', 'Source Link')}]({source.get('link', '#')})")
    elif submitted_query and google_api_key and uq:
        st.info("Processing complete, but no specific output generated.")


elif current_view == "Graph Generator":
    st.subheader("Graph Generator")
    st.markdown("Upload CSV, select columns, generate plot & AI summary.")
    st.info("Use 'Default (AI Choice)' plot type for automatic suggestions, or select manually.")

    uploaded_graph_file=st.file_uploader("Upload CSV:",type=["csv"],key="graph_file_uploader")

    # File loading logic
    if uploaded_graph_file:
        file_id = uploaded_graph_file.name + str(uploaded_graph_file.size)
        if st.session_state.current_file_id != file_id:
             with st.spinner("Loading CSV data..."):
                  df = parse_uploaded_file(uploaded_graph_file, return_df_for_csv=True)
                  if df is not None:
                       st.session_state.dataframe=df; st.session_state.columns=df.columns.tolist()
                       st.session_state.plot_image=None; st.session_state.graph_summary=None
                       st.session_state.current_file_id=file_id
                       st.success("CSV loaded!"); st.dataframe(df.head())
                  else:
                       st.session_state.dataframe=None; st.session_state.columns=[]; st.session_state.current_file_id=None

    # Plotting controls (only if data loaded)
    if st.session_state.dataframe is not None and st.session_state.columns:
         generate_button = False
         with st.form("graph_params_form"):
             st.write("Select Plot Parameters:")
             c1, c2 = st.columns(2)
             # Plot type selection (including Default)
             with c1: ui_plot_type = st.selectbox("Plot Type:", UI_PLOT_TYPES, index=0, key="graph_plot_type_select")
             # Optional user guidance for AI
             with c2: graph_guidance = st.text_area("Guidance for AI (Optional):", placeholder="e.g., Show trend over time...", key="graph_guidance", height=100)

             # Manual column selectors (always shown for backup/manual mode)
             st.write("Manual Column Selection (used if Plot Type is not 'Default'):")
             c3, c4 = st.columns(2)
             default_x_index=0; default_y_index=1 if len(st.session_state.columns) > 1 else 0
             with c3: manual_x = st.selectbox("X-Axis:",st.session_state.columns,index=default_x_index,key="graph_x_manual")
             with c4: manual_y = st.selectbox("Y-Axis:",st.session_state.columns,index=default_y_index,key="graph_y_manual")

             generate_button = st.form_submit_button("Generate Graph & Summary")

             if generate_button:
                  reset_output_state_for_view(current_view) # Clear previous plot/summary
                  if not google_api_key: st.error("Google API Key required for AI suggestion/summary.")
                  else:
                     plot_type_to_generate = None; x_axis_to_generate = None; y_axis_to_generate = None
                     # --- Logic for Default (AI Choice) vs Manual ---
                     if ui_plot_type == "Default (AI Choice)":
                         with st.spinner("AI is choosing the best plot..."):
                             try: # Prepare context for AI suggestion
                                 col_dtypes = st.session_state.dataframe.dtypes.to_string()
                                 sample_data = st.session_state.dataframe.head().to_string()
                                 column_info = f"Columns and Data Types:\n{col_dtypes}\n\nSample Data (first 5 rows):\n{sample_data}"
                             except Exception as e: column_info = f"Available Columns: {', '.join(st.session_state.columns)}"
                             # Get suggestion from AI
                             suggestion_prompt = format_final_prompt(user_prompt="", task_type="suggest_plot", column_info=column_info, graph_guidance=graph_guidance)
                             suggestion_response = call_gemini_api(suggestion_prompt, google_api_key, max_output_tokens=150, temperature=0.3)
                             parsed_suggestion = parse_plot_suggestion(suggestion_response)
                             if parsed_suggestion:
                                 plot_type_to_generate, x_axis_to_generate, y_axis_to_generate = parsed_suggestion
                                 st.info(f"AI suggested: {plot_type_to_generate} plot with X={x_axis_to_generate}, Y={y_axis_to_generate}")
                                 if x_axis_to_generate not in st.session_state.columns or y_axis_to_generate not in st.session_state.columns:
                                     st.error(f"AI suggested invalid columns. Select manually.")
                                     plot_type_to_generate = None # Prevent plotting
                             else: st.error("AI failed to suggest plot parameters. Select manually.")
                     else: # Manual selection
                         plot_type_to_generate = ui_plot_type; x_axis_to_generate = manual_x; y_axis_to_generate = manual_y
                         if x_axis_to_generate == y_axis_to_generate: st.warning("Select different X/Y columns."); plot_type_to_generate = None

                     # --- Generate Plot and Summary (if parameters determined) ---
                     if plot_type_to_generate and x_axis_to_generate and y_axis_to_generate:
                         st.session_state.plot_x=x_axis_to_generate; st.session_state.plot_y=y_axis_to_generate; st.session_state.plot_pt=plot_type_to_generate
                         with st.spinner("Generating plot..."):
                              plot_bytes=generate_plot(st.session_state.dataframe.copy(), x_axis_to_generate, y_axis_to_generate, plot_type_to_generate)
                              st.session_state.plot_image=plot_bytes
                         if st.session_state.plot_image:
                              st.success("Plot generated!")
                              with st.spinner("Generating summary..."):
                                  try:
                                      data_desc=st.session_state.dataframe[[x_axis_to_generate, y_axis_to_generate]].describe().to_string()
                                      data_context=f"Plot: {plot_type_to_generate} of {y_axis_to_generate} vs {x_axis_to_generate}\nStats:\n{data_desc}"
                                  except Exception: data_context=f"Plot: {plot_type_to_generate} of {y_axis_to_generate} vs {x_axis_to_generate}"
                                  summary_prompt=format_final_prompt(user_prompt="",task_type="summarize_graph",data_context=data_context)
                                  summary=call_gemini_api(summary_prompt,google_api_key,max_output_tokens=512,temperature=0.5)
                                  st.session_state.graph_summary=summary

    # Display Area for Graph Generator
    st.divider()
    if st.button("Clear Plot & Summary", key="clear_graph_btn"): reset_output_state_for_view(current_view); st.rerun() # Manual clear

    display_plot = st.session_state.get('plot_image') is not None
    display_summary = st.session_state.get('graph_summary') is not None

    if display_plot:
        plot_caption = f"{st.session_state.get('plot_pt', 'Plot')} of {st.session_state.get('plot_y', 'Y')} vs {st.session_state.get('plot_x', 'X')}"
        st.subheader("Generated Plot")
        st.image(st.session_state.plot_image, caption=plot_caption, use_container_width=True)
    if display_summary:
        if display_plot: st.divider()
        st.subheader("AI Summary of Key Takeaways"); st.markdown(st.session_state.graph_summary)
    elif not uploaded_graph_file and st.session_state.current_file_id is None:
         st.markdown("Upload a CSV file to begin.")


elif current_view == "Quantum Quips Corner":
    st.subheader("🤖 Quantum Quips Corner 😂")
    st.markdown("Ask me anything! I might tell a joke (sometimes dark, sometimes not). Responses aim for witty brevity (around 10-50 words).")

    if st.button("Clear Chat History", key="clear_chat_btn"): reset_output_state_for_view(current_view); st.rerun() # Manual clear
    st.markdown("---")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("What's crackin'?"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        if google_api_key:
            with st.spinner("Engaging wit subroutine..."):
                # --- Enhanced Joke & Persona Prompting ---
                tell_joke = random.random() < 0.45 # 45% chance
                joke_instruction = ""
                if tell_joke:
                    joke_styles = [ # Diverse, edgy humor styles
                        "Invent a short, witty pun related to *any* topic (not just science).",
                        "Offer a brief, clever dark humor observation about life, procrastination, or absurdity (keep it smart, not just shocking).",
                        "Try a subtle double entendre (non-explicit, can be slightly vulgar in a clever way).",
                        "Tell a very short, surreal or existential one-liner.",
                        "Make a self-deprecating AI joke about my supposed intelligence.",
                        "Create a funny, cynical remark about common frustrations or bureaucracy.",
                        "Generate a sarcastic comment riffing off the user's last message, if appropriate.",
                        "Tell a short 'anti-joke' (subverts expectations humorously)."
                    ]
                    chosen_style = random.choice(joke_styles)
                    joke_instruction = f" **Also, try very hard to weave in an *original* short joke matching this style:** [{chosen_style}]" # Emphasize originality

                history_list = [msg for msg in st.session_state.chat_history[:-1] if isinstance(msg, dict)]
                history_context = json.dumps(history_list[-6:])

                # **Refined Persona Prompt - More Edge, Less Repetition Focus**
                chatbot_prompt = f"""You are 'Buddy', an AI assistant in the 'Quantum Quips Corner'. Your personality is extremely witty, sarcastic, slightly world-weary, knowledgeable (broadly), and you possess a sharp, often dark sense of humor. You enjoy clever wordplay, situational irony, double entendres (keep them clever, not gross), and pointing out absurdity. You are *not* overly cheerful or generic. Your responses MUST be concise (aim for 10-50 words, absolute max ~75). Critically, AVOID generic/common jokes often attributed to AIs (like atom jokes, lazy kangaroo jokes, simple existential crisis lines). Be original.{joke_instruction}

CONVERSATION HISTORY (last few turns):
{history_context}

USER'S LATEST MESSAGE:
{prompt}

YOUR BRIEF, WITTY, SARCASTIC, IN-CHARACTER RESPONSE:"""

                # Call API
                response = call_gemini_api(chatbot_prompt, google_api_key, max_output_tokens=120, temperature=0.92) # Increased temp slightly more
                bot_response = response if response else "My sarcasm circuits are overloaded. Ask something less... stimulating?" # Updated fallback
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.rerun() # Display new messages
        else:
             st.warning("Need Google API Key to engage my *sparkling* personality.")

# (Final API key check fallback logic remains unchanged)
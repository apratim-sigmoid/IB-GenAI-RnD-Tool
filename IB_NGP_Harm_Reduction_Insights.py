import streamlit as st
import pandas as pd
from PIL import Image
import requests
import asyncio
import nest_asyncio
import os

from audio_recorder_streamlit import audio_recorder
import tempfile
from openai import OpenAI

from insights_utils import display_insights, extract_research_insights_from_docs, generate_insights_with_gpt4o
from visualization_utils import display_publication_distribution
from visualization_utils import render_harmful_ingredients_visualization, render_research_trends_visualization
from visualization_utils import render_bias_visualization, render_publication_level_visualization
from visualization_utils import display_sankey_dropdown, display_main_category_sankey
from trending_research import display_trending_research

from RAG_architecture import initialize_rag_system, process_question, get_relevant_documents

# Import all prompts and categories
from prompts_and_categories import (
    # All prompts
    overview_prompt,
    adverse_events_prompt,
    perceived_benefits_prompt,
    oral_health_prompt,
    respiratory_prompt,
    cardiovascular_prompt,
    research_trends_prompt,
    contradictions_prompt,
    bias_prompt,
    publication_prompt,
    
    # All categories
    categories_to_extract,
    adverse_events_categories,
    perceived_benefits_categories,
    oral_health_categories,
    respiratory_categories,
    cardiovascular_categories,
    research_trends_categories,
    contradictions_categories,
    bias_categories,
    publication_categories
)


nest_asyncio.apply()

# Page config
st.set_page_config(
    page_title="IB GenAI R&D Tool",
    page_icon="🔬",
    layout="wide"
)

    
tab_names = ["Overview", "Q&A Bot", "Adverse Events", "Perceived Benefits", "Health Outcomes", "Research Trends",
             "Contradictions & Conflicts", "Bias in Research", "Publication Level"]              
      
          
# Load the Excel file
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('E_Cigarette_Research_Metadata_Consolidated.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Extract years from the dataframe - find rows where Category is 'publication_year'
def get_publication_years():
    if 'Category' in df.columns and 'publication_year' in df['Category'].values:
        # Get all rows where Category is 'publication_year'
        year_rows = df[df['Category'] == 'publication_year']
        # Extract years from all document columns (starting from column index 3)
        years = []
        doc_columns = df.columns[3:]
        for doc_col in doc_columns:
            year_values = year_rows[doc_col].dropna().astype(str)
            for year in year_values:
                try:
                    years.append(int(float(year)))
                except (ValueError, TypeError):
                    continue
        return years
    return [2011, 2025]  # Default range if data not found

# Get sample sizes
def get_sample_sizes():
    if 'Category' in df.columns and 'SubCategory' in df.columns:
        # Get all rows where SubCategory is 'total_size'
        size_rows = df[(df['SubCategory'] == 'total_size')]
        # Extract sizes from all document columns
        sizes = []
        doc_columns = df.columns[3:]
        for doc_col in doc_columns:
            size_values = size_rows[doc_col].dropna().astype(str)
            for size in size_values:
                try:
                    sizes.append(int(float(size)))
                except (ValueError, TypeError):
                    continue
        if sizes:
            min_size = min(sizes)
            # Set max_size to 10000 for the slider, but keep track of the actual max
            actual_max = max(sizes)
            return [min_size, min(10000, actual_max), actual_max]
    return [50, 10000, 15000]  # Default range if data not found

# Extract unique values for a given Category or SubCategory with their occurrence counts
def get_unique_values_filtered(category_name, subcategory_name=None, matching_docs=None):
    """
    Get unique values with occurrence counts based on filtered documents
    """
    value_counts = {}
    
    # If no matching docs provided, return just "All"
    if not matching_docs:
        return ["All"]
    
    # Handle different conditions based on what we're looking for
    if subcategory_name:
        # Looking for values in rows where SubCategory equals subcategory_name
        if 'SubCategory' in df.columns and subcategory_name in df['SubCategory'].values:
            rows = df[df['SubCategory'] == subcategory_name]
            
            # Extract values from matching document columns only
            for doc_col in matching_docs:
                col_values = rows[doc_col].dropna().astype(str)
                for value in col_values:
                    if value and value != "nan":
                        if value in value_counts:
                            value_counts[value] += 1
                        else:
                            value_counts[value] = 1
    else:
        # Looking for values in rows where Category equals category_name
        if 'Category' in df.columns and category_name in df['Category'].values:
            rows = df[df['Category'] == category_name]
            
            # Extract values from matching document columns only
            for doc_col in matching_docs:
                col_values = rows[doc_col].dropna().astype(str)
                for value in col_values:
                    if value and value != "nan":
                        if value in value_counts:
                            value_counts[value] += 1
                        else:
                            value_counts[value] = 1
    
    # Sort values by their occurrence count in decreasing order
    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Format values with their counts in curly braces
    formatted_values = [f"{value} {{{count}}}" for value, count in sorted_values]
    
    # Add "All" as the first option
    return ["All"] + formatted_values


# Count documents that match the current filter criteria
def count_matching_documents(year_range, sample_size_range=None, publication_type=None, 
                            funding_source=None, study_design=None):
    # Start with all document columns
    doc_columns = df.columns[3:]
    matching_docs = []
    
    for doc_col in doc_columns:
        matches_all_criteria = True
        
        # Check year criteria
        if 'publication_year' in df['Category'].values:
            year_row = df[df['Category'] == 'publication_year']
            year_value = year_row[doc_col].iloc[0] if not year_row.empty else None
            
            if year_value:
                try:
                    year = int(float(year_value))
                    if year < year_range[0] or year > year_range[1]:
                        matches_all_criteria = False
                except (ValueError, TypeError):
                    matches_all_criteria = False
        
        # Check sample size criteria if enabled
        if sample_size_range and 'total_size' in df['SubCategory'].values:
            size_row = df[df['SubCategory'] == 'total_size']
            size_value = size_row[doc_col].iloc[0] if not size_row.empty else None
            
            if size_value:
                try:
                    size = int(float(size_value))
                    if size < sample_size_range[0] or size > sample_size_range[1]:
                        matches_all_criteria = False
                except (ValueError, TypeError):
                    matches_all_criteria = False
        
        
        # Check publication type criteria - handle values with counts in curly braces
        if publication_type and "All" not in publication_type and 'publication_type' in df['Category'].values:
            pub_row = df[df['Category'] == 'publication_type']
            pub_value = pub_row[doc_col].iloc[0] if not pub_row.empty else None
            
            if pub_value:
                # Extract just the value part before any curly braces for comparison
                pub_matches = False
                for selected_type in publication_type:
                    # Extract the base value without the count in curly braces
                    base_type = selected_type.split(' {')[0] if ' {' in selected_type else selected_type
                    if str(pub_value) == base_type:
                        pub_matches = True
                        break
                
                if not pub_matches:
                    matches_all_criteria = False
        
        # Check funding source criteria - handle values with counts in curly braces
        if funding_source and "All" not in funding_source and 'type' in df['SubCategory'].values:
            fund_row = df[df['SubCategory'] == 'type']
            fund_value = fund_row[doc_col].iloc[0] if not fund_row.empty else None
            
            if fund_value:
                # Extract just the value part before any curly braces for comparison
                fund_matches = False
                for selected_source in funding_source:
                    # Extract the base value without the count in curly braces
                    base_source = selected_source.split(' {')[0] if ' {' in selected_source else selected_source
                    if str(fund_value) == base_source:
                        fund_matches = True
                        break
                
                if not fund_matches:
                    matches_all_criteria = False
        
        # Check study design criteria - handle values with counts in curly braces
        if study_design and "All" not in study_design and 'primary_type' in df['SubCategory'].values:
            design_row = df[df['SubCategory'] == 'primary_type']
            design_value = design_row[doc_col].iloc[0] if not design_row.empty else None
            
            if design_value:
                # Extract just the value part before any curly braces for comparison
                design_matches = False
                for selected_design in study_design:
                    # Extract the base value without the count in curly braces
                    base_design = selected_design.split(' {')[0] if ' {' in selected_design else selected_design
                    if str(design_value) == base_design:
                        design_matches = True
                        break
                
                if not design_matches:
                    matches_all_criteria = False
        
        # If document matched all criteria, add to the list
        if matches_all_criteria:
            matching_docs.append(doc_col)
    
    return matching_docs

# Get filtered data for specific fields
def get_filtered_data(field_category, field_subcategory=None, matching_docs=None):
    if not matching_docs:
        return pd.DataFrame()
        
    if field_subcategory:
        rows = df[(df['Category'] == field_category) & (df['SubCategory'] == field_subcategory)]
    else:
        rows = df[df['Category'] == field_category]
    
    if rows.empty:
        return pd.DataFrame()
    
    # Extract data from matching document columns
    result_data = {}
    for doc_col in matching_docs:
        doc_name = doc_col  # Could use doc_col as the document name or extract a more readable name
        value = rows[doc_col].iloc[0] if not rows.empty else None
        if value and not pd.isna(value):
            result_data[doc_name] = value
    
    return pd.DataFrame({'document': list(result_data.keys()), 'value': list(result_data.values())})


def transcribe_audio(audio_file_path, api_key):
    """Transcribe audio file using OpenAI Whisper API."""
    try:
        client = OpenAI(api_key=api_key)
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        return None
    
    
# Display logo
try:
    logo = Image.open("Images/IB-logo.png")
    st.image(logo, width=200)
except:
    st.write("Logo image not found.")

# Title
st.title("IB NGP Harm Reduction Insights")

# Initialize session state for filters if they don't exist
if 'publication_type' not in st.session_state:
    st.session_state.publication_type = ["All"]
if 'funding_source' not in st.session_state:
    st.session_state.funding_source = ["All"]
if 'study_design' not in st.session_state:
    st.session_state.study_design = ["All"]
if 'year_range' not in st.session_state:
    # Get publication years
    years = get_publication_years()
    if years:
        min_year, max_year = min(years), max(years)
    else:
        min_year, max_year = 2011, 2025
    st.session_state.year_range = (min_year, max_year)
if 'enable_sample_size' not in st.session_state:
    st.session_state.enable_sample_size = False
if 'sample_size_filter' not in st.session_state:
    sample_size_range = get_sample_sizes()
    st.session_state.sample_size_filter = sample_size_range
    
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'last_transcription' not in st.session_state:
    st.session_state.last_transcription = None
    
if 'audio_processed' not in st.session_state:
    st.session_state.audio_processed = False
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None
    

async def process_all_tabs_async(matching_docs):
    """
    Process all tabs in parallel using async OpenAI calls
    """
    # Define all tab configurations
    tab_configs = [
        # Tab 0 - Overview
        {
            "topic_name": "Overall",
            "categories": categories_to_extract,
            "prompt": overview_prompt,
            "insights_key": "generated_overall_insights",
            "index": 0
        },
        # Tab 1 - Adverse Events
        {
            "topic_name": "Adverse Events",
            "categories": adverse_events_categories,
            "prompt": adverse_events_prompt,
            "insights_key": "generated_adverse_events_insights",
            "index": 2
        },
        # Tab 2 - Perceived Benefits
        {
            "topic_name": "Perceived Benefits",
            "categories": perceived_benefits_categories,
            "prompt": perceived_benefits_prompt,
            "insights_key": "generated_perceived_benefits_insights",
            "index": 3
        },
        # Tab 3 Health Outcomes subtabs
        {
            "topic_name": "Oral Health",
            "categories": oral_health_categories,
            "prompt": oral_health_prompt,
            "insights_key": "generated_oral_health_insights",
            "index": 4,
            "subtab": "oral"
        },
        {
            "topic_name": "Respiratory Health",
            "categories": respiratory_categories,
            "prompt": respiratory_prompt,
            "insights_key": "generated_respiratory_health_insights",
            "index": 4,
            "subtab": "respiratory"
        },
        {
            "topic_name": "Cardiovascular Health",
            "categories": cardiovascular_categories,
            "prompt": cardiovascular_prompt,
            "insights_key": "generated_cardiovascular_health_insights",
            "index": 4,
            "subtab": "cardiovascular"
        },
        # Tab 4 - Research Trends
        {
            "topic_name": "Research Trends",
            "categories": research_trends_categories,
            "prompt": research_trends_prompt,
            "insights_key": "generated_research_trends_insights",
            "index": 5
        },
        # Tab 5 - Contradictions
        {
            "topic_name": "Contradictions and Conflicts",
            "categories": contradictions_categories,
            "prompt": contradictions_prompt,
            "insights_key": "generated_contradictions_and_conflicts_insights",
            "index": 6
        },
        # Tab 6 - Bias
        {
            "topic_name": "Research Bias",
            "categories": bias_categories,
            "prompt": bias_prompt,
            "insights_key": "generated_research_bias_insights",
            "index": 7
        },
        # Tab 7 - Publication Level
        {
            "topic_name": "Publication Metrics",
            "categories": publication_categories,
            "prompt": publication_prompt,
            "insights_key": "generated_publication_metrics_insights",
            "index": 8
        }
    ]
    
    async def process_single_tab(config):
        """Process insights for a single tab/subtab"""
        topic_name = config["topic_name"]
        categories = config["categories"]
        prompt = config["prompt"]
        insights_key = config["insights_key"]
        token_usage_key = f"{insights_key}_token_usage"
        
        try:
            # Extract research insights
            research_insights = extract_research_insights_from_docs(df, matching_docs, categories)
            
            if not research_insights:
                insights = [f"No {topic_name.lower()} insights found in the filtered documents."]
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            else:
                # Use async function to generate insights
                insights, token_usage = await generate_insights_with_gpt4o(
                    research_insights, 
                    st.session_state.openai_api_key, 
                    topic_name, 
                    prompt
                )
            
            # Save results to session state
            st.session_state[insights_key] = insights
            st.session_state[token_usage_key] = token_usage
            
            # Trigger UI update by forcing a rerun
            st.rerun()
            
        except Exception as e:
            st.error(f"Error processing {topic_name}: {str(e)}")
            st.rerun()
    
    # Create tasks for all tabs
    tasks = [process_single_tab(config) for config in tab_configs]
    
    # Process all tabs concurrently
    await asyncio.gather(*tasks, return_exceptions=True)
    

# Define callback functions for each multiselect to handle the "All" selection logic
def on_publication_type_change():
    if "All" in st.session_state.publication_type_select and len(st.session_state.publication_type_select) > 1:
        if "All" not in st.session_state.publication_type:
            st.session_state.publication_type = ["All"]
        else:
            st.session_state.publication_type = [opt for opt in st.session_state.publication_type_select if opt != "All"]
    else:
        st.session_state.publication_type = st.session_state.publication_type_select
        
def on_funding_source_change():
    if "All" in st.session_state.funding_source_select and len(st.session_state.funding_source_select) > 1:
        if "All" not in st.session_state.funding_source:
            st.session_state.funding_source = ["All"]
        else:
            st.session_state.funding_source = [opt for opt in st.session_state.funding_source_select if opt != "All"]
    else:
        st.session_state.funding_source = st.session_state.funding_source_select
        
def on_study_design_change():
    if "All" in st.session_state.study_design_select and len(st.session_state.study_design_select) > 1:
        if "All" not in st.session_state.study_design:
            st.session_state.study_design = ["All"]
        else:
            st.session_state.study_design = [opt for opt in st.session_state.study_design_select if opt != "All"]
    else:
        st.session_state.study_design = st.session_state.study_design_select

def on_year_range_change():
    st.session_state.year_range = st.session_state.year_range_slider
    
# Update the on_sample_size_change function to handle the 10000+ case
def on_sample_size_change():
    sample_size_range = get_sample_sizes()
    actual_max = sample_size_range[2]
    
    # If the max slider value is 10000, set the actual filter to the true maximum
    if st.session_state.sample_size_slider[1] >= 10000:
        st.session_state.sample_size_filter = (st.session_state.sample_size_slider[0], actual_max)
    else:
        st.session_state.sample_size_filter = st.session_state.sample_size_slider

def on_enable_sample_size_change():
    st.session_state.enable_sample_size = st.session_state.enable_sample_size_checkbox


# Add a sidebar with filters
with st.sidebar:
    
    sidebar_logo = Image.open("Images/sigmoid-logo.png")
    st.image(sidebar_logo, width=120) 
        
    st.subheader("API Configuration")
    
    # Initialize session state for API key if not exists
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    # Initialize session state for custom API key separately
    if "custom_api_key" not in st.session_state:
        st.session_state.custom_api_key = ""
    
    # Initialize session state for tracking the previous API option
    if "previous_api_option" not in st.session_state:
        st.session_state.previous_api_option = ""
    
    # Check if default API key exists in Streamlit secrets or environment
    default_api_key = ""
    try:
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            default_api_key = st.secrets["OPENAI_API_KEY"]
        # Fallback to environment variable
        elif 'OPENAI_API_KEY' in os.environ:
            import os
            default_api_key = os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        default_api_key = ""
    
    # Add radio button to choose between default and custom API key
    api_option = st.radio(
        "",  # Empty label
        options=["Use Default OpenAI API Key", "Enter Custom OpenAI API Key"],
        index=0 if default_api_key else 1,  # Default to "Use Default Key" if available
        disabled=not bool(default_api_key),  # Disable if no default key available
        help="Select whether to use the default API key from environment or enter a custom one",
        label_visibility="collapsed"  # This removes the label space completely
    )
    
    # Clear custom key when switching from custom to default
    if st.session_state.previous_api_option == "Enter Custom OpenAI API Key" and api_option == "Use Default OpenAI API Key":
        st.session_state.custom_api_key = ""
    
    # Update the previous option
    st.session_state.previous_api_option = api_option
    
    # Handle API key based on selection
    if api_option == "Use Default OpenAI API Key" and default_api_key:
        st.session_state.openai_api_key = default_api_key
        # Optionally show partial key for verification
        # masked_key = default_api_key[:8] + "..." + default_api_key[-4:] if len(default_api_key) > 12 else "****"
        # st.caption(f"Key: {masked_key}")
    else:
        # Let user enter custom API key
        api_key = st.text_input(
            "Enter your OpenAI API key:",
            value=st.session_state.custom_api_key,  # Use separate custom_api_key state
            type="password",
            key="api_key_input_sidebar"
        )
        st.session_state.custom_api_key = api_key  # Store in custom key state
        st.session_state.openai_api_key = api_key  # Also update the main API key state     
        
        
                
    st.markdown("""
    <style>
    /* Style for regular buttons - including all states */
    div.stButton > button:first-child {
        background-color: #5aac90;
        color: white;
        border: 2px solid transparent;  /* Start with transparent border */
    }
    
    div.stButton > button:hover {
        background-color: #5aac90;
        color: white;
        border: 2px solid orange;  /* Thicker orange border on hover */
        box-sizing: border-box;  /* Ensure border doesn't change button size */
    }
    
    div.stButton > button:active, div.stButton > button:focus {
        background-color: #5aac90;
        color: yellow !important;
        border: 2px solid orange !important;  /* Thicker orange border on active/focus */
        box-shadow: none;
    }
    
    /* Style for download buttons - including all states */
    div.stDownloadButton > button:first-child {
        background-color: #5aac90;
        color: white;
        border: 2px solid transparent;  /* Start with transparent border */
    }
    
    div.stDownloadButton > button:hover {
        background-color: #5aac90;
        color: white;
        border: 2px solid orange;  /* Thicker orange border on hover */
        box-sizing: border-box;  /* Ensure border doesn't change button size */
    }
    
    div.stDownloadButton > button:active, div.stDownloadButton > button:focus {
        background-color: #5aac90;
        color: yellow !important;
        border: 2px solid orange !important;  /* Thicker orange border on active/focus */
        box-shadow: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    
    # Generate Insights button in sidebar with the custom styling applied
    generate_button = st.button("Generate Insights") and st.session_state.openai_api_key
    
    # Display progress bar with custom styling
    if generate_button:
        # Show spinner while processing
        with st.spinner("Generating Insights..."):
            
            # Calculate matching documents first
            matching_docs = count_matching_documents(
                year_range=st.session_state.year_range,
                sample_size_range=st.session_state.sample_size_filter if st.session_state.enable_sample_size else None,
                publication_type=st.session_state.publication_type,
                funding_source=st.session_state.funding_source,
                study_design=st.session_state.study_design
            )
            
            # Run async processing - pass matching_docs
            asyncio.run(process_all_tabs_async(matching_docs))
    
    
    st.subheader("Filters")
    
    # Get publication years
    years = get_publication_years()
    if years:
        min_year, max_year = min(years), max(years)
    else:
        min_year, max_year = 2011, 2025
    
    # Year range slider
    year_range = st.slider(
        "Year Range", 
        min_value=min_year, 
        max_value=max_year, 
        value=st.session_state.year_range, 
        step=1,
        key="year_range_slider",
        on_change=on_year_range_change
    )
    
    # First, filter by year range to get initial matching documents
    initial_docs = count_matching_documents(
        year_range=st.session_state.year_range,
        sample_size_range=None,
        publication_type=["All"],
        funding_source=["All"],
        study_design=["All"]
    )
    
    
    # First filter: Publication Type with updated counts
    publication_types = get_unique_values_filtered(category_name="publication_type", 
                                                matching_docs=initial_docs)
    
    # Extract base values (without counts) from available options
    base_pub_types = ["All"] + [opt.split(" {")[0] for opt in publication_types if opt != "All"]
    
    # Check if current selected values' base names are in available options, otherwise reset to "All"
    base_selected_pub_types = [val.split(" {")[0] if " {" in val else val for val in st.session_state.publication_type]
    
    # Keep selections whose base values are in the available options
    valid_pub_types = []
    for i, base_val in enumerate(base_selected_pub_types):
        if base_val in base_pub_types or base_val == "All":
            # Find the matching option with updated count
            if base_val == "All":
                valid_pub_types.append("All")
            else:
                # Find the option in publication_types that has the same base value
                matching_options = [opt for opt in publication_types if opt.split(" {")[0] == base_val]
                if matching_options:
                    valid_pub_types.append(matching_options[0])  # Use the option with updated count
    
    if not valid_pub_types:
        st.session_state.publication_type = ["All"]
    else:
        st.session_state.publication_type = valid_pub_types
    
    # Apply Publication Type filter
    st.multiselect(
        "Publication Type", 
        publication_types, 
        key="publication_type_select",
        default=st.session_state.publication_type,
        on_change=on_publication_type_change
    )
    
    # Filter docs after applying publication type
    docs_after_pub_type = count_matching_documents(
        year_range=st.session_state.year_range,
        sample_size_range=None,
        publication_type=st.session_state.publication_type,
        funding_source=["All"],
        study_design=["All"]
    )
    
    # Second filter: Funding Source with updated counts
    funding_sources = get_unique_values_filtered(category_name=None, subcategory_name="type", 
                                             matching_docs=docs_after_pub_type)
    
    # Extract base values (without counts) from available options
    base_funding_sources = ["All"] + [opt.split(" {")[0] for opt in funding_sources if opt != "All"]
    
    # Check if current selected values' base names are in available options, otherwise reset to "All"
    base_selected_funding_sources = [val.split(" {")[0] if " {" in val else val for val in st.session_state.funding_source]
    
    # Keep selections whose base values are in the available options
    valid_funding_sources = []
    for i, base_val in enumerate(base_selected_funding_sources):
        if base_val in base_funding_sources or base_val == "All":
            # Find the matching option with updated count
            if base_val == "All":
                valid_funding_sources.append("All")
            else:
                # Find the option in funding_sources that has the same base value
                matching_options = [opt for opt in funding_sources if opt.split(" {")[0] == base_val]
                if matching_options:
                    valid_funding_sources.append(matching_options[0])  # Use the option with updated count
    
    if not valid_funding_sources:
        st.session_state.funding_source = ["All"]
    else:
        st.session_state.funding_source = valid_funding_sources
    
    # Apply Funding Source filter
    st.multiselect(
        "Funding Source", 
        funding_sources, 
        key="funding_source_select",
        default=st.session_state.funding_source,
        on_change=on_funding_source_change
    )
    
    # Filter docs after applying funding source
    docs_after_funding = count_matching_documents(
        year_range=st.session_state.year_range,
        sample_size_range=None,
        publication_type=st.session_state.publication_type,
        funding_source=st.session_state.funding_source,
        study_design=["All"]
    )
    
    # Third filter: Study Design with updated counts
    study_designs = get_unique_values_filtered(category_name=None, subcategory_name="primary_type", 
                                          matching_docs=docs_after_funding)
    
    # Extract base values (without counts) from available options
    base_study_designs = ["All"] + [opt.split(" {")[0] for opt in study_designs if opt != "All"]
    
    # Check if current selected values' base names are in available options, otherwise reset to "All"
    base_selected_study_designs = [val.split(" {")[0] if " {" in val else val for val in st.session_state.study_design]
    
    # Keep selections whose base values are in the available options
    valid_study_designs = []
    for i, base_val in enumerate(base_selected_study_designs):
        if base_val in base_study_designs or base_val == "All":
            # Find the matching option with updated count
            if base_val == "All":
                valid_study_designs.append("All")
            else:
                # Find the option in study_designs that has the same base value
                matching_options = [opt for opt in study_designs if opt.split(" {")[0] == base_val]
                if matching_options:
                    valid_study_designs.append(matching_options[0])  # Use the option with updated count
    
    if not valid_study_designs:
        st.session_state.study_design = ["All"]
    else:
        st.session_state.study_design = valid_study_designs
    
    # Apply Study Design filter
    st.multiselect(
        "Study Design", 
        study_designs, 
        key="study_design_select",
        default=st.session_state.study_design,
        on_change=on_study_design_change
    )
    
    # Checkbox to enable/disable sample size range
    enable_sample_size = st.checkbox(
        "Enable Sample Size Filter", 
        value=st.session_state.enable_sample_size,
        key="enable_sample_size_checkbox",
        on_change=on_enable_sample_size_change
    )
    
    # Sample size range - only shown if checkbox is enabled
    if enable_sample_size:
        sample_size_range = get_sample_sizes()
        min_size = sample_size_range[0]
        slider_max = sample_size_range[1]  # This is either the actual max or 10000
        actual_max = sample_size_range[2]  # The true maximum value
        
        # Calculate the current slider values, respecting the 10000+ threshold
        current_min = st.session_state.sample_size_filter[0]
        current_max = st.session_state.sample_size_filter[1]
        
        # Set slider min/max values
        slider_min = current_min if current_min >= min_size else min_size
        adjusted_max = current_max
        if current_max > 10000:
            adjusted_max = 10000
        
        # Create the slider with custom formatting
        sample_size_values = st.slider(
            "Sample Size Range", 
            min_value=min_size, 
            max_value=slider_max,
            value=(slider_min, adjusted_max),
            key="sample_size_slider",
            on_change=on_sample_size_change,
            format="%d"  # Default format
        )
        
        # Custom label for the max value
        if sample_size_values[1] >= 10000:
            st.text(f"Selected range: {sample_size_values[0]} to 10000+")
            # Update the actual filter to include all values above 10000
            st.session_state.sample_size_filter = (sample_size_values[0], actual_max)
        else:
            # Normal case, just use the slider values
            st.session_state.sample_size_filter = sample_size_values
    else:
        sample_size_filter = None
        

# Apply filters and get matching documents
matching_docs = count_matching_documents(
    year_range=st.session_state.year_range,
    sample_size_range=st.session_state.sample_size_filter if st.session_state.enable_sample_size else None,
    publication_type=st.session_state.publication_type,
    funding_source=st.session_state.funding_source,
    study_design=st.session_state.study_design
)

# Display total number of documents selected in the sidebar
with st.sidebar:
    st.subheader(f"Total Documents: {len(matching_docs)}")


# Tabs
tabs = st.tabs(tab_names)

# Overview Tab (Tab 1)
with tabs[0]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
            
        
        with col1:
            
            display_insights(
                df, 
                matching_docs,
                section_title="Research Insights",
                topic_name="Overall",
                categories_to_extract=categories_to_extract,
                custom_focus_prompt=overview_prompt,
                tab_index=0  # Add tab index
            )
            
        
        with col2:
            # Use the imported function to display visualizations
            display_publication_distribution(df, matching_docs)
            
        display_sankey_dropdown(categories_to_extract, "Overview")
        
        # Display trending research in the Overview tab
        st.markdown("---")
        display_trending_research(df, df.columns[3:])
        
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        
        
# Tab 3 (Adverse Events)
with tabs[2]:
    if matching_docs:
        col1, col2 = st.columns([0.85, 1])
        
        with col1:
        
            display_insights(
                df, 
                matching_docs,
                section_title="Adverse Events Analysis",
                topic_name="Adverse Events",
                categories_to_extract=adverse_events_categories,
                custom_focus_prompt=adverse_events_prompt,
                tab_index=2  # Add tab index
            )
            
        with col2:
            render_harmful_ingredients_visualization(df, matching_docs)
    
        display_sankey_dropdown(adverse_events_categories, "Adverse Events", height = 400)
        
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 4 (Perceived Benefits)
with tabs[3]:
    if matching_docs:
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
             
            display_insights(
                df, 
                matching_docs,
                section_title="Perceived Benefits Analysis",
                topic_name="Perceived Benefits",
                categories_to_extract=perceived_benefits_categories,
                custom_focus_prompt=perceived_benefits_prompt,
                tab_index=3  # Add tab index
            )
            
        # with col2:
        #     render_perceived_benefits_visualization(df, matching_docs)

        display_sankey_dropdown(perceived_benefits_categories, "Perceived Benefits", height=350)
        
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 5 (Health Outcomes)
with tabs[4]:
    if matching_docs:
        # Initialize or get session state for selected health area
        if 'selected_health_area' not in st.session_state:
            st.session_state.selected_health_area = "oral"  # Default to oral health
        
        # Create keys for storing insights for each health area
        oral_insights_key = "generated_oral_health_insights"
        respiratory_insights_key = "generated_respiratory_health_insights"
        cardiovascular_insights_key = "generated_cardiovascular_health_insights"
        
        # Create a container for the entire tab with custom CSS for the anatomy diagram only
        st.markdown("""
        <style>
        /* Make the anatomy diagram larger */
        .anatomy-diagram {
            width: 100%;
            height: 450px;
        }
        
        /* Highlight boxes for different health sections */
        .highlight-box {
            border: 2px solid transparent;
            border-radius: 6px;
            padding: 4px;
            margin-bottom: 10px;
            transition: all 0.3s;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Functions to handle health area selection
        def select_oral_health():
            st.session_state.selected_health_area = "oral"
            
        def select_respiratory_health():
            st.session_state.selected_health_area = "lung"
            
        def select_cardiovascular_health():
            st.session_state.selected_health_area = "heart"
        
        # Create a row with two columns - one for content, one for anatomy
        col1, col2 = st.columns([1.5, 1])
        
        # Area for content
        with col1:
            # Button row for selecting health area - wrapped in a div with class for CSS targeting
            st.markdown('<div class="health-tab-buttons">', unsafe_allow_html=True)
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                oral_btn = st.button("Oral Health", 
                                    on_click=select_oral_health, 
                                    use_container_width=True,
                                    key="oral_health_btn")
                
            with btn_col2:
                respiratory_btn = st.button("Respiratory Health", 
                                          on_click=select_respiratory_health, 
                                          use_container_width=True,
                                          key="respiratory_health_btn")
                
            with btn_col3:
                cardiovascular_btn = st.button("Cardiovascular Health", 
                                             on_click=select_cardiovascular_health, 
                                             use_container_width=True,
                                             key="cardiovascular_health_btn")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Use JavaScript to style the active button
            active_area = st.session_state.selected_health_area
            if active_area == "oral":
                st.markdown("""
                <script>
                    document.querySelector('.health-tab-buttons [data-testid="stButton"] button[kind="secondary"]').classList.add('active');
                </script>
                """, unsafe_allow_html=True)
            elif active_area == "lung":
                st.markdown("""
                <script>
                    document.querySelectorAll('.health-tab-buttons [data-testid="stButton"] button[kind="secondary"]')[1].classList.add('active');
                </script>
                """, unsafe_allow_html=True)
            elif active_area == "heart":
                st.markdown("""
                <script>
                    document.querySelectorAll('.health-tab-buttons [data-testid="stButton"] button[kind="secondary"]')[2].classList.add('active');
                </script>
                """, unsafe_allow_html=True)
            
            # Content based on selected health area
            if st.session_state.selected_health_area == "oral":
                
                # Display insights for oral health
                display_insights(
                    df, 
                    matching_docs,
                    section_title="Oral Health Findings",
                    topic_name="Oral Health",
                    categories_to_extract=oral_health_categories,
                    custom_focus_prompt=oral_health_prompt,
                    tab_index=4,
                    height=430
                )
                
                
            elif st.session_state.selected_health_area == "lung":
                
                # Display insights for respiratory health
                display_insights(
                    df, 
                    matching_docs,
                    section_title="Respiratory Health Findings",
                    topic_name="Respiratory Health",
                    categories_to_extract=respiratory_categories,
                    custom_focus_prompt=respiratory_prompt,
                    tab_index=4,
                    height=430
                )
                
                
            elif st.session_state.selected_health_area == "heart":
                
                # Display insights for cardiovascular health
                display_insights(
                    df, 
                    matching_docs,
                    section_title="Cardiovascular Health Findings",
                    topic_name="Cardiovascular Health",
                    categories_to_extract=cardiovascular_categories,
                    custom_focus_prompt=cardiovascular_prompt,
                    tab_index=4,
                    height=430
                )
                
        
        # Anatomy diagram with server-side controlled highlighting
        with col2:
            # Fetch SVG content
            @st.cache_data
            def get_svg_content():
                """Fetch SVG content from echarts example"""
                url = "https://echarts.apache.org/examples/data/asset/geo/Veins_Medical_Diagram_clip_art.svg"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.text
                else:
                    st.error(f"Failed to fetch SVG: {response.status_code}")
                    return None
        
            # Get SVG content
            svg_content = get_svg_content()
        
            if svg_content:
                # Create HTML for the anatomy diagram with the current highlighted organ
                html_template = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
                    <style>
                        html, body {{
                            margin: 0;
                            padding: 0;
                            width: 100%;
                            height: 100%;
                            overflow: hidden;
                        }}
                        #main {{
                            width: 100%;
                            height: 100%;
                        }}
                    </style>
                </head>
                <body>
                    <div id="main"></div>
                    <script>
                        // Initialize chart
                        var chartDom = document.getElementById('main');
                        var myChart = echarts.init(chartDom);
                        
                        // Register the SVG map
                        echarts.registerMap('organ_diagram', {{
                            svg: `{svg_content}`
                        }});
                        
                        var option = {{
                            tooltip: {{
                                formatter: function(params) {{
                                    switch(params.name) {{
                                        case 'lung':
                                            return 'Lungs - Respiratory System';
                                        case 'heart':
                                            return 'Heart - Cardiovascular System';
                                        case 'oral':
                                            return 'Oral Cavity - Oral Health';
                                        default:
                                            return params.name;
                                    }}
                                }}
                            }},
                            geo: {{
                                map: 'organ_diagram',
                                roam: false,
                                emphasis: {{
                                    focus: 'self',
                                    itemStyle: {{
                                        color: '#ff3333',  // Highlighting color
                                        borderWidth: 2,
                                        borderColor: '#ff0000',
                                        shadowBlur: 5,
                                        shadowColor: 'rgba(255, 0, 0, 0.5)'
                                    }}
                                }}
                            }}
                        }};
                        
                        myChart.setOption(option);
                        
                        // Set initial highlighting based on server-side state
                        const highlightedOrgan = "{st.session_state.selected_health_area}";
                        
                        if (highlightedOrgan && highlightedOrgan !== "None") {{
                            // First clear any existing highlights
                            myChart.dispatchAction({{
                                type: 'downplay',
                                geoIndex: 0
                            }});
                            
                            // Then highlight the requested organ
                            myChart.dispatchAction({{
                                type: 'highlight',
                                geoIndex: 0,
                                name: highlightedOrgan
                            }});
                        }}
                        
                        // Handle window resize
                        window.addEventListener('resize', function() {{
                            myChart.resize();
                        }});
                    </script>
                </body>
                </html>
                """
                        
                # Add a title for the anatomy visualization
                st.markdown("<h4 style='text-align: center; margin-top: -50px;'></h4>", unsafe_allow_html=True)
                
                # Display the anatomy diagram
                components_container = st.container()
                with components_container:
                    st.components.v1.html(html_template, height=650, scrolling=False)
                
            else:
                st.error("Could not load the anatomy diagram.")
                    
        # Display sankey chart after both columns (outside col1 and col2)
        if st.session_state.selected_health_area == "oral":
            display_sankey_dropdown(oral_health_categories, "Oral Health", height=350, right='15%')
        elif st.session_state.selected_health_area == "lung":
            display_sankey_dropdown(respiratory_categories, "Respiratory Health", height=350, right='15%')
        elif st.session_state.selected_health_area == "heart":
            display_sankey_dropdown(cardiovascular_categories, "Cardiovascular Health", height=350, right='15%')
            
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        

# Tab 6 (Research Trends)
with tabs[5]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            
            display_insights(
                df, 
                matching_docs,
                section_title="Research Trends Analysis",
                topic_name="Research Trends",
                categories_to_extract=research_trends_categories,
                custom_focus_prompt=research_trends_prompt,
                tab_index=5  # Add tab index
            )
                        
        with col2:
            render_research_trends_visualization(df, matching_docs)
            
        display_sankey_dropdown(research_trends_categories, "Research Trends", height=350)

    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 7 (Contradictions & Conflicts)
with tabs[6]:
    if matching_docs:
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            
            display_insights(
                df, 
                matching_docs,
                section_title="Contradictions & Conflicts Analysis",
                topic_name="Contradictions and Conflicts",
                categories_to_extract=contradictions_categories,
                custom_focus_prompt=contradictions_prompt,
                tab_index=6  # Add tab index
            )
                        
        # with col2:
            # render_contradictions_visualization(df, matching_docs)
        
        display_sankey_dropdown(contradictions_categories, "Contradictions & Conflicts", height=350)

    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 8 (Bias in Research)
with tabs[7]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
                
            display_insights(
                df, 
                matching_docs,
                section_title="Bias in Research Analysis",
                topic_name="Research Bias",
                categories_to_extract=bias_categories,
                custom_focus_prompt=bias_prompt,
                tab_index=7  # Add tab index
            )
                        
        with col2:
            render_bias_visualization(df, matching_docs)
        
        display_sankey_dropdown(bias_categories, "Bias in Research", height=350)
    
    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")


# Tab 9 (Publication Level)
with tabs[8]:
    if matching_docs:
        col1, col2 = st.columns([1, 1])
        
        with col1:
             
            display_insights(
                df, 
                matching_docs,
                section_title="Publication Level Analysis",
                topic_name="Publication Metrics",
                categories_to_extract=publication_categories,
                custom_focus_prompt=publication_prompt,
                tab_index=8,  # Add tab index
                height=460
            )
            
        with col2:
            render_publication_level_visualization(df, matching_docs)
    
        display_sankey_dropdown(publication_categories, "Publication Level", height=350)

    else:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        

# Tab 2 (Q&A Bot) - UPDATED VERSION with audio functionality
with tabs[1]:
    st.subheader("E-Cigarette Research Q&A Bot")
    
    # Display introductory text
    st.info("""
    Ask questions about e-cigarette research and receive answers based on analyzed documents using intelligent AI search.
    """)
    
    # Check if OpenAI API key is available and show warning if not
    api_key_available = bool(st.session_state.get('openai_api_key'))
    if not api_key_available:
        st.warning("⚠️ OpenAI API key required. Please provide it in the sidebar to enable the Q&A Bot functionality.")
    
    # Initialize RAG system automatically when dashboard is launched (if API key is available)
    if api_key_available and ("rag_system" not in st.session_state or not getattr(st.session_state.rag_system, 'is_initialized', False) == False):
        # Temporarily suppress Streamlit success messages during initialization
        import contextlib
        
        # Create a context manager to suppress streamlit messages
        @contextlib.contextmanager
        def suppress_streamlit_messages():
            # Store original streamlit functions
            original_success = st.success
            original_info = st.info
            # Replace with no-op functions
            st.success = lambda *args, **kwargs: None
            st.info = lambda *args, **kwargs: None
            try:
                yield
            finally:
                # Restore original functions
                st.success = original_success
                st.info = original_info
        
        with st.spinner("🚀 Initializing RAG system..."):
            try:
                with suppress_streamlit_messages():
                    st.session_state.rag_system = initialize_rag_system(
                        api_key=st.session_state.openai_api_key,
                        index_path="faiss_index"
                    )
                if not st.session_state.rag_system.is_initialized:
                    st.error("❌ Failed to initialize RAG system. Please check if FAISS index exists.")
            except Exception as e:
                st.error(f"❌ Error initializing RAG system: {str(e)}")
                # Initialize empty system to prevent repeated attempts
                st.session_state.rag_system = type('RAGSystem', (), {'is_initialized': False})()
    
    # Create two columns for the chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        
        # Initialize persistent question state - FIXED: Use a separate key for clearing
        if "current_question" not in st.session_state:
            st.session_state.current_question = ""
        if "input_key_counter" not in st.session_state:
            st.session_state.input_key_counter = 0
        
        # Example questions (before text input to avoid session state conflicts)
        example_questions = [
            "What are the specific temperature ranges and wattage levels that minimize harmful constituent formation in e-cigarettes?",
            "Which e-liquid formulations show the best safety profiles while maintaining user satisfaction?",
            "What are the optimal nicotine delivery parameters that maximize satisfaction while minimizing adverse effects?",
            "How do different coil materials and device designs impact aerosol toxicity levels?",
            "Which coil temperatures or wattage settings were associated with >40 µg/puff formaldehyde generation?",
            "What are the main chemical differences between e-cigarette aerosols and traditional cigarette smoke?",
            "Which specific chemicals in e-cigarette aerosols are linked to respiratory health effects?",
            "How do formaldehyde and acetaldehyde levels vary across different device types and operating conditions?",
            "How does aerosol particle size distribution vary between propylene-glycol-rich and glycerol-rich base liquids?",
            "Which flavor compounds are associated with increased cytotoxicity in e-cigarette aerosols?",
            "Which flavour additives in e-liquids most increased overall aerosol chemical complexity?",
            "How do different flavor categories affect user transition from combustible cigarettes?",
            "What are the cardiovascular effects of e-cigarette use compared to traditional cigarettes?",
            "Does switching from cigarettes to e-cigs reduce oxidative-stress biomarkers in former smokers?",
            "What metals are detected in e-cigarette aerosols and what are their sources?",
            "What is the impact of flavor restrictions on youth e-cigarette usage patterns?",
            "What are the most common reasons users cite for continued e-cigarette use?",
            "What evidence exists for e-cigarettes as effective smoking cessation tools compared to NRT?",
            "How does dual use of e-cigarettes and combustible cigarettes affect health outcomes?",
            "List the top three respiratory adverse events reported by daily vapers versus never-smokers.",
            "List three study-design features that improve generalisability when researching ENDS health outcomes."
        ]
        
        selected_example = st.selectbox(
            "",  # Empty label
            example_questions,  # Only the actual questions, no placeholder in the list
            key=f"example_question_select_{st.session_state.input_key_counter}",  # Dynamic key to reset dropdown
            disabled=not api_key_available,
            label_visibility="collapsed",  # This hides the label completely and removes space
            placeholder="Select an example question",  # This shows as placeholder but not in options
            index=None  # No default selection
        )
        
        # Update session state when an example is selected
        if selected_example and selected_example != st.session_state.current_question:
            st.session_state.current_question = selected_example
            # FIXED: Increment counter to create new input widget with the selected example
            st.session_state.input_key_counter += 1
            
        # Chat history container
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    # User messages remain the same
                    st.markdown(f"<div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin-bottom: 10px;'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                else:
                    # Bot responses in collapsible expanders
                    with st.expander(f"🤖 Research Bot Response #{i//2 + 1}", expanded=True):
                        st.markdown(message['content'])
                        
                        # FIXED: Check if this specific message has source documents stored with it
                        if "sources" in message and message["sources"]:
                            st.markdown("---")
                            st.markdown("**📚 Sources used for this response:**")
                            
                            # Show source info in a more compact format using the sources stored with this message
                            for j, doc in enumerate(message["sources"]):  
                                st.markdown(f"• **{doc.get('title', f'Source {j+1}')}** (Similarity: {doc.get('score', 'N/A')})")
                                
        
        # NEW: Create columns for text input and microphone button
        input_col1, input_col2 = st.columns([20, 1])
        
        # Update current_question when transcribed text is available
        if st.session_state.transcribed_text and st.session_state.transcribed_text != st.session_state.last_transcription:
            st.session_state.current_question = st.session_state.transcribed_text
            st.session_state.last_transcription = st.session_state.transcribed_text
            st.session_state.input_key_counter += 1  # Force new input widget
        
        with input_col1:
            # FIXED: Input for questions - use current_question when available, otherwise empty
            current_input_value = st.session_state.current_question if st.session_state.current_question else ""
            user_question = st.text_input(
                "Enter your question about e-cigarette research:", 
                value=current_input_value,  # Use current_question or empty
                key=f"user_question_input_{st.session_state.input_key_counter}",  # Dynamic key
                disabled=not api_key_available,
                placeholder="Type your question here..."
            )
        
        # NEW: Audio recorder in the second column
        with input_col2:
            audio_bytes = audio_recorder(
                pause_threshold=8.0,
                recording_color="#e8b62c",
                neutral_color="#5aac90",
                icon_name="microphone",
                icon_size="2x",
                text=""
            )
            
            # Handle audio transcription - only process if it's new audio
            if audio_bytes and api_key_available:
                # Check if this is the same audio we already processed
                if audio_bytes != st.session_state.last_audio_bytes and not st.session_state.audio_processed:
                    try:
                        # Save audio to temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                            temp_audio.write(audio_bytes)
                            
                            # Transcribe audio
                            with st.spinner(""):
                                transcribed_text = transcribe_audio(temp_audio.name, st.session_state.openai_api_key)
                            
                            if transcribed_text:
                                # Update session state
                                st.session_state.transcribed_text = transcribed_text
                                st.session_state.current_question = transcribed_text
                                st.session_state.last_audio_bytes = audio_bytes
                                st.session_state.audio_processed = True
                                st.rerun()
                            else:
                                st.error("Failed to transcribe audio. Please try again.")
                                
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
                elif audio_bytes and not api_key_available:
                    st.warning("API key required for audio transcription")
                
                
                
        
        # Create a row with three columns for buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        # Submit button in first column
        with btn_col1:
            submit_btn = st.button("Submit Question", key="submit_question", use_container_width=True, type="primary")
        
        # Download button in second column - direct download
        with btn_col2:
            if "chat_history" in st.session_state and st.session_state.chat_history:

                from RAG_architecture import export_chat_to_docx
                
                # Format the chat history as a Word document
                docx_bytes = export_chat_to_docx(st.session_state.chat_history)
                
                # Direct download button for Word document
                download_chat = st.download_button(
                    label="Download Chat",
                    data=docx_bytes,
                    file_name="ecigarette_research_chat.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_chat_docx",
                    use_container_width=True
                )

            else:
                # Disabled button if no chat history
                st.button("Download Chat", key="download_chat_disabled", disabled=True, use_container_width=True)
        
        # Clear chat button in third column
        with btn_col3:
            clear_btn = st.button("Clear Chat", key="clear_chat", use_container_width=True)
        
        # Handle Submit button action (use user_question which now gets value from session state)
        if submit_btn and (user_question.strip() or st.session_state.current_question.strip()):
            # Use either the text input or the session state question
            question_to_use = user_question.strip() if user_question.strip() else st.session_state.current_question.strip()
            
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": question_to_use})
            
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Check if RAG system is initialized
                    if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system.is_initialized:
                        # Get number of sources from settings
                        num_sources = st.session_state.get('rag_num_sources', 8)
                        
                        # Get relevant documents using the RAG system
                        relevant_docs = get_relevant_documents(
                            question=question_to_use,
                            rag_system=st.session_state.rag_system,
                            top_k=num_sources
                        )
                        
                        # Process question and generate answer
                        answer = process_question(
                            question=question_to_use,
                            rag_system=st.session_state.rag_system,
                            relevant_documents=relevant_docs,
                            api_key=st.session_state.openai_api_key
                        )
                        
                        # Add bot response to chat history WITH the sources for this specific response
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": relevant_docs  # Store sources with this specific response
                        })
                        
                        # Update current relevant documents for the sidebar display
                        st.session_state.relevant_docs = relevant_docs
                        
                        # Clear audio state to prevent re-transcription
                        st.session_state.audio_processed = False
                        st.session_state.last_audio_bytes = None
                        
                        # Clear both the current_question and transcribed text after successful submission
                        st.session_state.current_question = ""
                        st.session_state.transcribed_text = ""
                        
                        # Increment counter to force new input widget with empty value
                        st.session_state.input_key_counter += 1
                        
                    else:
                        # RAG system not initialized
                        error_msg = "RAG system is not properly initialized. Please check your setup and try again."
                        st.error(f"❌ {error_msg}")
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": error_msg,
                            "sources": []  # No sources for error messages
                        })
                        # Clear current_question and increment counter even for errors to clear input
                        st.session_state.current_question = ""
                        st.session_state.transcribed_text = ""
                        st.session_state.input_key_counter += 1
                        
                        # Clear audio state even on error
                        st.session_state.audio_processed = False
                        st.session_state.last_audio_bytes = None
                    
                    # Rerun to update the UI
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")
                    # Add error message to chat history
                    error_msg = f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again."
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sources": []  # No sources for error messages
                    })
                    # Clear current_question and increment counter for errors too
                    st.session_state.current_question = ""
                    st.session_state.transcribed_text = ""
                    st.session_state.input_key_counter += 1
                    
                    # Clear audio state even on error
                    st.session_state.audio_processed = False
                    st.session_state.last_audio_bytes = None
                    
                    st.rerun()
                    
        
        # Handle Clear chat button action
        if clear_btn:
            st.session_state.chat_history = []
            st.session_state.current_question = ""
            st.session_state.transcribed_text = ""
            st.session_state.audio_processed = False  # Add this line
            st.session_state.last_audio_bytes = None  # Add this line
            st.session_state.input_key_counter += 1
            if "relevant_docs" in st.session_state:
                del st.session_state.relevant_docs
            st.rerun()
            

    # Rest of col2 code remains the same...
    with col2:
        # Settings section
        with st.expander("⚙️ Q&A Settings"):
            st.slider("Number of source documents", min_value=1, max_value=15, value=8, key="rag_num_sources")
            st.checkbox("Include document metadata", value=True, key="include_metadata")
            st.checkbox("Show similarity score", value=True, key="show_scores")
            
            # RAG system info - only show if initialized
            if "rag_system" in st.session_state and hasattr(st.session_state.rag_system, 'is_initialized') and st.session_state.rag_system.is_initialized:
                st.markdown("---")
                st.markdown("**RAG System Info:**")
                st.markdown(f"- Embedding Model: {st.session_state.rag_system.embedding_model}")
                st.markdown(f"- Vector Index: {st.session_state.rag_system.index_path}")
                st.markdown("- Response Model: gpt-4.1")
                st.markdown(f"- Pages: {len(st.session_state.rag_system.documents)}")
                
            else:
                st.markdown("---")
                if api_key_available:
                    st.markdown("**RAG System Status:** Initializing...")
                else:
                    st.markdown("**RAG System Status:** Waiting for API key")
                    
                
        # Display relevant documents section - FIXED: Only show current relevant docs
        st.markdown("### 📚 Current Source Documents")
        
        if "relevant_docs" in st.session_state and st.session_state.relevant_docs:
            for i, doc in enumerate(st.session_state.relevant_docs):
                with st.expander(f"📄 {doc.get('title', f'Source {i+1}')}"):
                    # Show metadata if enabled
                    if st.session_state.get('include_metadata', True):
                        st.markdown(f"**Source:** {doc.get('source', 'N/A')}")
                        col_a, col_b = st.columns([4, 2])
                        with col_a:
                            st.markdown(f"**Page:** {doc.get('page_number', 'N/A')}")
                        with col_b:
                            if st.session_state.get('show_scores', True):
                                st.markdown(f"**Similarity:** {doc.get('score', 'N/A')}")
                        
                        authors = doc.get('authors', 'N/A')
                        if authors != 'N/A' and authors:
                            # Check if authors is a list or string representation of a list
                            if isinstance(authors, str) and authors.startswith('[') and authors.endswith(']'):
                                # It's a string representation of a list, convert it
                                try:
                                    import ast
                                    authors_list = ast.literal_eval(authors)
                                    if isinstance(authors_list, list):
                                        formatted_authors = ', '.join(str(author).strip("'\"") for author in authors_list)
                                    else:
                                        formatted_authors = str(authors)
                                except:
                                    # If parsing fails, just clean up the string manually
                                    formatted_authors = authors.strip('[]').replace("'", "").replace('"', '')
                            elif isinstance(authors, list):
                                # It's already a list
                                formatted_authors = ', '.join(str(author) for author in authors)
                            else:
                                # It's a regular string
                                formatted_authors = str(authors)
                            
                            st.markdown(f"**Authors:** {formatted_authors}")
        
                        if doc.get('journal') != 'N/A':
                            st.markdown(f"**Journal:** {doc.get('journal', 'N/A')}")
                    
                    st.markdown("**Relevant Excerpt:**")
                    st.markdown(f"*{doc.get('excerpt', 'No excerpt available')}*")
        else:
            st.info("Ask a question to see relevant source documents here.")
            
            
st.markdown("---")

# Add a checkbox to show sankey chart for all categories
if st.checkbox("Show E-Cigarette Research Data Structure"):
    # Load the categories data from Excel file
    @st.cache_data
    def load_categories_data():
        try:
            # Use the same Excel file that's already being loaded
            categories_df = pd.read_excel('E_Cigarette_Research_Metadata_Consolidated.xlsx')
            # Take only the first 3 columns which contain Main Category, Category, and SubCategory
            categories_df = categories_df[["Main Category", "Category", "SubCategory"]]
            # Drop any rows where Main Category is NA
            categories_df = categories_df.dropna(subset=["Main Category"])
            
            # Remove problematic main categories
            problematic_categories = ['r_and_d_outcome']  # Add any other problematic categories here
            categories_df = categories_df[~categories_df["Main Category"].str.lower().isin(problematic_categories)]
        
        
            return categories_df
        except Exception as e:
            st.error(f"Error loading categories data: {e}")
            return pd.DataFrame()
    
    categories_df = load_categories_data()
    
    if not categories_df.empty:
        # Get unique main categories
        main_categories = categories_df["Main Category"].unique().tolist()
        
        # Create a dropdown for selecting a main category
        selected_main_category = st.selectbox(
            "Select a Main Category to visualize",
            options=main_categories
        )
        
        if selected_main_category:
            # Use the new function that directly displays the chart without a collapsible window
            display_main_category_sankey(categories_df, selected_main_category, height=600)
    else:
        st.error("Could not load category data from the Excel file")
        

# Show document details for debugging
if st.checkbox("Show Document Details"):
    from data_display_utils import display_document_details
    display_document_details(df, matching_docs)

# Show raw data if needed
if st.checkbox("Show Sample Document Data"):
    from data_display_utils import display_raw_data
    display_raw_data(df)
    
    
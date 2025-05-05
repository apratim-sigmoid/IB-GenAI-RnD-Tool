import streamlit as st
import pandas as pd
from functools import lru_cache
from openai import AsyncOpenAI



# Define the tab names for the progress tracking
tab_names = ["Overview", "Adverse Events", "Perceived Benefits", "Health Outcomes", 
             "Research Trends", "Contradictions & Conflicts", "Bias in Research", "Publication Level"]

def extract_research_insights_from_docs(df, matching_docs, categories_to_extract):
    """
    Extract comprehensive research insights from matching documents using custom categories.
    Only includes non-missing attributes to provide better context.
    
    Args:
        df (DataFrame): The dataframe containing all research data
        matching_docs (list): List of document columns that match filter criteria
        categories_to_extract (dict, optional): Dictionary of categories and subcategories to extract
                                               If None, uses default categories
    
    Returns:
        dict: Structured insights data organized by document and category
    """
    insights = {}
    
    # For each matching document, extract the insights
    for doc_col in matching_docs:
        doc_insights = {}
        
        # Get title if available
        title = None
        title_row = df[(df['Main Category'] == 'meta_data') & (df['Category'] == 'title')]
        if not title_row.empty:
            title = title_row[doc_col].iloc[0]
        if title is None:
            title_row = df[df['Category'] == 'title']
            if not title_row.empty:
                title = title_row[doc_col].iloc[0]
        
        doc_identifier = title if title and not pd.isna(title) else doc_col
        
        # Process each main category
        for main_category, subcategories in categories_to_extract.items():
            category_insights = {}
            
            for subcategory in subcategories:
                # Look for exact matches first
                subcategory_rows = df[df['Category'] == subcategory]
                
                # If not found, try partial matches
                if subcategory_rows.empty:
                    subcategory_rows = df[df['Category'].str.contains(subcategory, na=False)]
                
                # If still not found, look for it in SubCategory
                if subcategory_rows.empty and 'SubCategory' in df.columns:
                    subcategory_rows = df[df['SubCategory'] == subcategory]
                    
                    if subcategory_rows.empty:
                        subcategory_rows = df[df['SubCategory'].str.contains(subcategory, na=False)]
                
                if not subcategory_rows.empty:
                    subcategory_data = subcategory_rows[doc_col].dropna().tolist()
                    # Only include non-empty data
                    if subcategory_data and any(str(item).strip() != "" for item in subcategory_data):
                        category_insights[subcategory] = subcategory_data
            
            # Only include categories with actual data
            if category_insights:
                doc_insights[main_category] = category_insights
        
        # Only include documents with actual insights
        if doc_insights:
            insights[doc_identifier] = doc_insights
    
    return insights


async def generate_insights_with_gpt4o(insights_data, api_key, topic_name="Research", custom_focus_prompt=None):
    """
    Pass the extracted research insights to GPT-4o and get concise bullet point insights.
    
    Args:
        insights_data (dict): Structured insights data organized by document and category
        api_key (str): OpenAI API key
        topic_name (str): The name of the topic for prompt customization
        custom_focus_prompt (str, optional): Custom prompt section for specific focus areas
        
    Returns:
        tuple: (list of generated bullet points with insights, dict with token usage information)
    """
    if not insights_data:
        return [f"No {topic_name.lower()} insights found in the filtered documents."], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    try:
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Format the structured insights data into a readable text format for the prompt with improved context
        formatted_insights = []
        
        for doc_id, doc_data in insights_data.items():
            formatted_insights.append(f"DOCUMENT: {doc_id}")
            
            for category, category_data in doc_data.items():
                # Add category header only if there's actual data
                if category_data:
                    formatted_insights.append(f"\n{category}:")
                    
                    for subcategory, values in category_data.items():
                        # Skip empty values
                        if not values or all(pd.isna(v) for v in values) or all(str(v).strip() == "" for v in values):
                            continue
                            
                        # Create a human-readable version of the subcategory by replacing dots and underscores
                        readable_subcategory = subcategory.replace('.', ' → ').replace('_', ' ').title()
                        
                        if isinstance(values, list):
                            # For lists, prefix each value with its meaning
                            if len(values) == 1:
                                formatted_insights.append(f"  - {readable_subcategory}: {values[0]}")
                            else:
                                formatted_insights.append(f"  - {readable_subcategory}:")
                                for i, val in enumerate(values):
                                    if str(val).strip():  # Only include non-empty values
                                        formatted_insights.append(f"      * Value {i+1}: {val}")
                        else:
                            if str(values).strip():  # Only include non-empty values
                                formatted_insights.append(f"  - {readable_subcategory}: {values}")
                        
            formatted_insights.append("\n---\n")
        
        # Prepare the prompt with specific formatting instructions
        prompt = f"""
        You are an expert researcher analyzing e-cigarette and vaping studies. Below are detailed {topic_name.lower()} insights from several studies, organized by document and category. 
        
        Based on these insights, generate 7-10 concise, insightful bullet points that capture the key findings, patterns, and implications across the studies.
        
        {custom_focus_prompt}
        
        IMPORTANT FORMATTING INSTRUCTION:
        - Use ONLY a single bullet point character '•' at the beginning of each insight
        - DO NOT use any secondary or nested bullet points
        - DO NOT start any line with any other bullet character or symbol
        
        Focus on precise measurements, numerical values, and specific technical details that directly enable product improvement.

        Here are the {topic_name.lower()} insights:
        
        {'\n'.join(formatted_insights)}
        
        Please respond with only the bullet points, each starting with a '•' character.
        """
        
        # Make API call to GPT-4.1
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that generates concise {topic_name.lower()} insights with simple bullet points. Never use nested bullet points. Always clearly indicate what metrics and units are being used."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        
        # Extract token usage information
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        
        # Extract and process the bullet points
        insights_text = response.choices[0].message.content
        
        # Split the text into bullet points, making sure each starts with •
        bullet_points = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line and line.startswith('•'):
                # Remove any potential nested bullets by replacing any bullet characters
                # that might appear after the initial bullet with their text equivalent
                clean_line = line.replace(' • ', ': ')  # Replace nested bullets with colons
                bullet_points.append(clean_line)
            elif line and bullet_points:  # For lines that might be continuation of previous bullet point
                # Make sure there are no bullet characters in continuation lines
                clean_line = line.replace('•', '')
                bullet_points[-1] += ' ' + clean_line
        
        # If no bullet points were found with •, try to parse by lines
        if not bullet_points:
            bullet_points = [line.strip().replace('•', '') for line in insights_text.split('\n') if line.strip()]
        
        return bullet_points, token_usage
    
    except Exception as e:
        return [f"Error generating {topic_name.lower()} insights: {str(e)}"], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    

# Add a cache for API responses
@lru_cache(maxsize=32)
def cached_generate_insights(insights_data_str, api_key, topic_name, custom_focus_prompt):
    """Cached version of the generate_insights function to avoid duplicate API calls"""
    # Convert insights_data_str back to dictionary
    import json
    insights_data = json.loads(insights_data_str)
    insights, token_usage = generate_insights_with_gpt4o(insights_data, api_key, topic_name, custom_focus_prompt)
    # Return both the insights and token usage
    return insights, token_usage


def display_insights(df, matching_docs, section_title="Research Insights", 
                     topic_name="Research", categories_to_extract=None, 
                     custom_focus_prompt=None,
                     wordcloud_path="Images/ecigarette_research_wordcloud.png",
                     enable_throttling=False,  # Disabled for async processing
                     tab_index=-1, height=525):
    """
    Displays insights for a single tab with async processing support.
    Note: Categories and prompts for Tab 3 subtabs are handled in process_all_tabs_async().
    """
    st.subheader(section_title)
    
    # Check if there are matching documents
    if not matching_docs:
        st.warning("No documents match the selected filters. Please adjust your filter criteria.")
        return
    
    # Create container for scrollable content
    insights_container = st.container()
    
    with insights_container:
        # Key for storing insights in session state
        insights_key = f"generated_{topic_name.lower().replace(' & ', '_').replace(' ', '_')}_insights"
        token_usage_key = f"{insights_key}_token_usage"
        
        # For health outcomes tab, handle based on the selected health area
        if tab_index == 3:
            # Tab 3 has subtabs - determine which one to display based on selection
            health_area = st.session_state.get('selected_health_area', 'oral')
            
            if health_area == "oral":
                insights_key = "generated_oral_health_insights"
                token_usage_key = f"{insights_key}_token_usage"
            elif health_area == "lung":
                insights_key = "generated_respiratory_health_insights"
                token_usage_key = f"{insights_key}_token_usage"
            elif health_area == "heart":
                insights_key = "generated_cardiovascular_health_insights"
                token_usage_key = f"{insights_key}_token_usage"
        
        # Display generated insights if available
        if insights_key in st.session_state:
            # Display previously generated insights with direct height styling
            insights_html = f'<div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem;">'
            for insight in st.session_state[insights_key]:
                insights_html += f"<p>{insight}</p>"
            
            # Add token usage information at the bottom if available
            if token_usage_key in st.session_state:
                token_usage = st.session_state[token_usage_key]
                token_limit = 1000000  # GPT-4.1 token limit
                token_percentage = (token_usage["total_tokens"] / token_limit) * 100
                insights_html += f"<p style='font-size: 0.8em; color: #666; border-top: 1px solid #ddd; padding-top: 5px;'>Tokens used: {token_usage['total_tokens']} ({token_percentage:.1f}% of 1 million token limit)</p>"
            
            insights_html += "</div>"
            st.markdown(insights_html, unsafe_allow_html=True)
            
        else:
            # Empty state with wordcloud and direct height styling
            message = f"Click the 'Generate Insights' button to analyze {topic_name.lower()} findings."
            
            # Determine appropriate message based on state
            if not st.session_state.openai_api_key:
                message = "Please enter your OpenAI API key to generate insights."
            else:
                message = f"Click the 'Generate Insights' button to analyze {topic_name.lower()} findings."
            
            try:
                # Load the wordcloud image
                import base64
                with open(wordcloud_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                html = f"""
                <div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <p style="color: #666; text-align: left; margin-bottom: 0px; position: absolute; top: 8px; left: 20px; right: 0; z-index: 2;">{message}</p>
                    <img src="data:image/png;base64,{encoded_image}" style="width: 100%; height: 100%; object-fit: cover; padding: 35px 0px 15px 0px;" />
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div style="height: {height}px; overflow-y: auto; padding: 0.5rem; border: 2px solid #f8d6d5; border-radius: 0.5rem; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <p style="color: #666; text-align: center;">{message}</p>
                    <p style="color: #999; font-size: 0.8em;">Unable to load wordcloud image: {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
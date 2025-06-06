import streamlit as st
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
from pandasai.smart_dataframe import SmartDataframe

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better formatting
st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stChatMessage [data-testid="stChatMessageContent"] {
        padding: 1rem;
    }
    .stChatMessage [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.5rem;
    }
    .stChatMessage [data-testid="stChatMessageContent"] pre {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stChatMessage [data-testid="stChatMessageContent"] code {
        background-color: #f0f2f6;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def plot_and_display_chart(df, prompt):
    """Function to create and display charts based on the data and user query"""
    try:
        # Set the style for seaborn
        sns.set_style("whitegrid")
        
        # Create a figure with a larger size
        plt.figure(figsize=(12, 6))
        
        # Get column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Determine plot type based on user query
        prompt_lower = prompt.lower()
        
        if "bar" in prompt_lower or "barplot" in prompt_lower:
            if len(categorical_cols) > 0:
                # Create bar plot for categorical data
                if len(numeric_cols) > 0:
                    # If we have numeric columns, use the first one for the y-axis
                    sns.barplot(x=categorical_cols[0], y=numeric_cols[0], data=df)
                    plt.title(f'Bar Plot of {numeric_cols[0]} by {categorical_cols[0]}')
                else:
                    # If no numeric columns, create a count plot
                    sns.countplot(x=categorical_cols[0], data=df)
                    plt.title(f'Count of {categorical_cols[0]}')
            else:
                st.warning("No categorical columns found for bar plot")
                return
                
        elif "pie" in prompt_lower or "pie chart" in prompt_lower:
            if len(categorical_cols) > 0:
                # Create pie chart for categorical data
                if len(numeric_cols) > 0:
                    # If we have numeric columns, use the first one for values
                    values = df.groupby(categorical_cols[0])[numeric_cols[0]].sum()
                else:
                    # If no numeric columns, use counts
                    values = df[categorical_cols[0]].value_counts()
                
                plt.pie(values, labels=values.index, autopct='%1.1f%%')
                plt.title(f'Pie Chart of {categorical_cols[0]}')
            else:
                st.warning("No categorical columns found for pie chart")
                return
                
        elif "line" in prompt_lower or "lineplot" in prompt_lower:
            if len(numeric_cols) >= 2:
                # Create line plot for numeric data
                sns.lineplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
                plt.title(f'Line Plot of {numeric_cols[1]} vs {numeric_cols[0]}')
            else:
                st.warning("Need at least two numeric columns for line plot")
                return
                
        elif "scatter" in prompt_lower or "scatterplot" in prompt_lower:
            if len(numeric_cols) >= 2:
                # Create scatter plot for numeric data
                sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1])
                plt.title(f'Scatter Plot of {numeric_cols[1]} vs {numeric_cols[0]}')
            else:
                st.warning("Need at least two numeric columns for scatter plot")
                return
                
        elif "hist" in prompt_lower or "histogram" in prompt_lower:
            if len(numeric_cols) > 0:
                # Create histogram for numeric data
                sns.histplot(data=df, x=numeric_cols[0], kde=True)
                plt.title(f'Distribution of {numeric_cols[0]}')
            else:
                st.warning("No numeric columns found for histogram")
                return
                
        elif "box" in prompt_lower or "boxplot" in prompt_lower:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                # Create box plot
                sns.boxplot(x=categorical_cols[0], y=numeric_cols[0], data=df)
                plt.title(f'Box Plot of {numeric_cols[0]} by {categorical_cols[0]}')
            else:
                st.warning("Need both numeric and categorical columns for box plot")
                return
                
        elif "heatmap" in prompt_lower or "correlation" in prompt_lower:
            if len(numeric_cols) >= 2:
                # Create correlation heatmap
                correlation = df[numeric_cols].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
            else:
                st.warning("Need at least two numeric columns for correlation heatmap")
                return
                
        else:
            # Default to correlation heatmap if no specific plot type is requested
            if len(numeric_cols) >= 2:
                correlation = df[numeric_cols].corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Heatmap')
            else:
                st.warning("No specific plot type requested and insufficient data for default plot")
                return
        
        # Rotate x-axis labels if they're too long
        plt.xticks(rotation=45, ha='right')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Close the plot to free memory
        plt.close()
        
    except Exception as e:
        st.error(f"Error creating the plot: {str(e)}")

def format_response(response):
    """Format the response to be more readable"""
    try:
        if isinstance(response, pd.DataFrame):
            return response
        elif isinstance(response, (list, tuple)):
            return "\n".join([f"- {item}" for item in response])
        elif isinstance(response, dict):
            return "\n".join([f"**{key}**: {value}" for key, value in response.items()])
        elif response is None:
            return "No results found."
        else:
            # Convert to string and clean up
            response_str = str(response)
            # Remove any special characters or formatting issues
            response_str = response_str.replace("'", "").replace('"', '')
            return response_str
    except Exception as e:
        return f"Error formatting response: {str(e)}"

def chat_with_csv(df, prompt):
    """Function to handle chat with CSV using PandasAI"""
    try:
        llm = OpenAI(id="gpt-4o", api_token=openai_api_key)
        
        # Enhanced prompt for better responses
        enhanced_prompt = f"""
        Please analyze the data and provide a clear, well-structured response to the following question:
        {prompt}
        
        Guidelines for your response:
        1. Start with a brief summary of your findings
        2. Use clear, concise language
        3. Format numbers and statistics appropriately
        4. If showing calculations, explain the steps
        5. If creating visualizations, use seaborn for plotting
        6. End with key takeaways or recommendations
        
        For list queries (like 'what are all item names'), please return a simple list of items.
        For statistical queries, please include both the numbers and their interpretation.
        
        Please structure your response in a clear, organized manner.
        """
        
        pandas_ai = SmartDataframe(
            df, 
            config={
                "llm": llm, 
                "save_charts": False,
                "verbose": True,
                "enforce_privacy": True,
                "enable_cache": True,
                "use_error_correction_framework": True,
                "max_retries": 3,
                "custom_instructions": enhanced_prompt
            }
        )
        
        # Handle specific types of queries
        if "what are all" in prompt.lower() or "list all" in prompt.lower():
            # For listing queries, try to get a simple list
            try:
                # Get unique values from the relevant column
                column_name = prompt.lower().split("all")[-1].strip().replace("names", "name").replace("items", "item")
                if column_name in df.columns:
                    result = df[column_name].unique().tolist()
                    return format_response(result)
            except Exception as e:
                st.warning(f"Could not get simple list, falling back to full analysis: {str(e)}")
        
        result = pandas_ai.chat(prompt)
        
        # Check if the prompt is about plotting
        if any(word in prompt.lower() for word in ["plot", "graph", "chart", "visualize", "show"]):
            plot_and_display_chart(df, prompt)
            
        return format_response(result)
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Sidebar for file upload and API key
with st.sidebar:
    st.header("⚙️ Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read the file based on its extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                df = None
                
            if df is not None:
                st.session_state['df'] = df
                st.success(f"Successfully uploaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Title and description with image
st.markdown('<h1><img src="https://static.wixstatic.com/media/97b8c3_d0a1a2e3860e436fbc5712b8c33c65f9~mv2.gif" width="60" height="55" style="vertical-align: middle;"> Data Analysis Chatbot</h1>', unsafe_allow_html=True)
st.markdown("Ask questions about your data and get instant insights!")

# Main content area
if 'df' in st.session_state:
    df = st.session_state['df']
    
    # Display data preview at the top using expander
    with st.expander("📋 Click to view Data Preview", expanded=True):
        st.dataframe(df, use_container_width=True)
    
    # Add a separator
    st.markdown("---")
    
    # Chat interface below
    st.subheader("💬 Chat with your Data")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"], use_container_width=True)
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                try:
                    response = chat_with_csv(df, prompt)
                    if isinstance(response, pd.DataFrame):
                        st.dataframe(response, use_container_width=True)
                    else:
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("👈 Please upload a CSV file using the sidebar to begin analysis.")

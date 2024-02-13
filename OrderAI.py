import streamlit as st
import os
import pandas as pd
from datetime import datetime
import json
import io
from PIL import Image
# utility and ai functions from utils_ai.py
from utils_ai import json_gpt, query_embeddings_and_search, transcribe_audio, extract_text_from_image, extract_text_from_image_llava

st.set_page_config(page_title='OrderAI')

def app():
    """Main function that runs the Streamlit app."""
    st.markdown(
        "<h2 style='text-align: center;'>OrderAI</h2>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # initialize button variables before the tabs
    text_order_button = False
    audio_order_button = False
    image_order_button = False

    tab1, tab2, tab3 = st.tabs(["Text", "Audio", "Image"])

    with tab1:
        order = st.text_area(
        "Enter an order",
        "",
        )
        if len(order) > 0:
            text_order_button = st.button("Run text analysis", type="primary")
    
    with tab2:
        uploaded_audio_file = st.file_uploader("Upload an audio file", type=['mp3', 'wav', 'ogg'])
        if uploaded_audio_file:
            st.audio(uploaded_audio_file, format='audio/wav', start_time=0)
            audio_order_button = st.button("Run audio analysis", type="primary")

    with tab3:
        uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg'])
        if uploaded_image:
            image_bytes = uploaded_image.getvalue()
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, width=400)
            image_model = st.selectbox(
            'Model',
            ('GPT-4-Vision', 'LLaVA-v1.6-34b'))
            image_order_button = st.button("Run image analysis", type="primary")        
    
    st.markdown("<br>", unsafe_allow_html=True)

    if text_order_button or audio_order_button or image_order_button:
        st.divider()

        if audio_order_button:
            with st.spinner(f"Transcribing audio... \n\n"):
                transcript = transcribe_audio(uploaded_audio_file)
                if transcript:
                    st.markdown(f"<h4 style='text-align: center;'>Transcript</h4>", unsafe_allow_html=True)
                    st.markdown(transcript)
                    st.markdown("<br>", unsafe_allow_html=True)
                    # assign the transcript value to order
                    order = transcript

        if image_order_button:
            with st.spinner(f"Extracting text from image... \n\n"):
                if image_model == "GPT-4-Vision":
                    extraction = extract_text_from_image(uploaded_image)
                elif image_model == "LLaVA-v1.6-34b":
                    extraction = extract_text_from_image_llava(uploaded_image)
                if extraction:
                    st.markdown(f"<h4 style='text-align: center;'>Extraction</h4>", unsafe_allow_html=True)
                    st.markdown(extraction)
                    st.markdown("<br>", unsafe_allow_html=True)
                    # assign the extraction value to order
                    order = extraction

        try:
            with st.spinner(f"Analyzing the order... \n\n"):
                
                # extract order info in json structure
                json_output = json_gpt(order)
                order_data = json.loads(json_output)
                
                # normalize quantity to int or float
                for item in order_data["order"]["items"]:
                    quantity_str = str(item["quantity"])
                    item["quantity"] = float(quantity_str) if '.' in quantity_str else int(quantity_str)

                # create df from items
                items_df = pd.DataFrame(order_data["order"]["items"])

                # embedding and semantic search for each item name in catalog vector database
                results = [query_embeddings_and_search(name) for name in items_df['name']]

                # add new columns to df
                for col in ['Supplier', 'Item', 'Code', 'Score']:
                    items_df[col] = ""

                # process search results and update df
                for index, search_result in enumerate(results):
                    top_match = search_result['matches'][0] if search_result['matches'] else None
                    if top_match:
                        items_df.at[index, 'Supplier'] = top_match['metadata']['supplier']  # Get supplier from top match
                        items_df.at[index, 'Item'] = top_match['metadata']['name']
                        items_df.at[index, 'Code'] = top_match['metadata']['code']
                        items_df.at[index, 'Score'] = round(top_match['score'], 2)

                # check if there's only one unique supplier
                unique_suppliers = items_df['Supplier'].unique()

                # remove unnecessary columns and update column names
                items_df.drop(columns=['code', 'name'], inplace=True)  # Adjust based on your actual unnecessary columns
                items_df.rename(columns={'unit': 'Unit', 'quantity': 'Quantity', 'Score': '- Score -'}, inplace=True)
                items_df = items_df[['Supplier', 'Code', 'Item', 'Unit', 'Quantity', '- Score -']]
                # remove supplier as well if unique supplier
                if len(unique_suppliers) == 1:
                    items_df.drop(columns=['Supplier'], inplace=True)
                
                st.markdown(f"<h4 style='text-align: center;'>Order</h4>", unsafe_allow_html=True)
                if len(unique_suppliers) == 1:
                    st.markdown(f"<h5>Order items</h5>", unsafe_allow_html=True)
                    st.data_editor(items_df, num_rows="dynamic")  # Assuming st.data_editor is your method to display/edit data
                else:
                    # split and display data frame for each supplier
                    for supplier in unique_suppliers:
                        supplier_df = items_df[items_df['Supplier'] == supplier].reset_index(drop=True)
                        supplier_df = supplier_df[['Supplier', 'Code', 'Item', 'Unit', 'Quantity', '- Score -']]  # Reorder for clarity
                        st.markdown(f"<h5>From {supplier}</h5>", unsafe_allow_html=True)
                        st.data_editor(supplier_df, num_rows="dynamic")

                # display delivery information
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>Delivery</h5>", unsafe_allow_html=True)
                datetime_str = order_data.get("order", {}).get("delivery", {}).get("datetime")
                if datetime_str:
                    try:
                        delivery_datetime = datetime.fromisoformat(datetime_str)
                        st.date_input("Delivery Date", delivery_datetime.date())
                        st.time_input("Delivery Time", delivery_datetime.time())
                    except ValueError:
                        st.error("Delivery datetime is missing or is not provided in the correct format.")
                else:
                    st.warning("Delivery datetime is missing or is not provided in the correct format.")

                # display comment
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>Comment</h5>", unsafe_allow_html=True)
                comment = order_data.get("order", {}).get("comment", "No comment.")
                st.markdown(comment if comment else "No comment.", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()

                # more details explanation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align: center;'>Details on the process</h4>", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>0. Prerequisite: Having supplier catalogs embedded in a vector database</h5>", unsafe_allow_html=True)
                st.markdown("Embed catalog items with an embedding model (here 1536 dimensions with OpenAI's text-embedding-3-small) and store the resulting vectors and their metadata (supplier, label, code, category...) in a vector index, such as Pinecone.", unsafe_allow_html=True)
                st.image('media/schema_part1.png', caption='Catalog embedding and vector storage')

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>1. Transcript of the order if non-text format</h5>", unsafe_allow_html=True)
                st.markdown("""
                For non-text input, first transform the order into a string:
                            
                - To transcribe audio files: use Whisper, speech-to-text open source model developed by OpenAI.
                            
                - To extract text from images: use GPT-4-Vision, state-of-the-art multimodal model from OpenAI currently in preview, or LLaVA v1.6 34b, one of the best open source models today to extract information from images.
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>2. Extraction of the order information in JSON format using GPT-4</h5>", unsafe_allow_html=True)
                st.markdown('Define a JSON structure and force GPT to return a response in this format, by writing in the system message "Output only valid JSON" and by defining well the prompt.', unsafe_allow_html=True)
                st.markdown('JSON is a useful format to exchange data between applications and GPTs are pretty good at writing it.', unsafe_allow_html=True)
                st.code(json_output, "json")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>3. Embedding of the order items and semantic search in vector database</h5>", unsafe_allow_html=True)
                st.markdown('For each item in the order, embed it and perform a semantic search in the Pinecone index using cosine similarity to find the most similar items in the catalog and return their metadata (label, code, category...) to fill in the order management UI.', unsafe_allow_html=True)
                st.markdown('If the resulting items are from different suppliers, the order is split.', unsafe_allow_html=True)
                st.image('media/schema_part2.png', caption='Querying the vector index')
                st.markdown("Semantic search uses vector representations of text (text placed in ta multi-dimensional space depending on its meaning) to interpret query intent and document relevance beyond keyword matching.", unsafe_allow_html=True)
                st.markdown("Cosine similarity measures the similarity between two vectors in a multi-dimensional space, using the cosine of the angle between two vectors and determining whether they are pointing in the same direction.", unsafe_allow_html=True)
                st.image('media/schema_cosine.png', caption='Cosine similarity simplified illustration')

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<h5>Potential improvements</h5>", unsafe_allow_html=True)
                st.markdown("""
                There are some potential improvements, including:
                            
                - Use a hybrid search combining keyword search and semantic search, and add a check on the item code if was specified, for more control and potentially better results.
                            
                - Handle more complex cases such as ambiguous delivery instructions like "tomorrow" or when the same item is in different supplier catalogs.
                            
                - To reduce costs and reduce platform dependency, we could use open source models, such as multilingual-E5-large for embedding or Mixtral-8x7B as the LLM. They are performant enough for the task, even though GPT-4 ensures additional capture of complex information.
                """, unsafe_allow_html=True)
            

        except Exception as e:
            st.write(
                "Error during the process. Please reload below and try again."
            )
            st.write(
                {e}
            )
            if st.button("Reload"):
                st.experimental_rerun()
            st.stop()


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    app()


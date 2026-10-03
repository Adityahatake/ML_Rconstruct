import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq


# -------------------- Setup --------------------
load_dotenv() 


@st.cache_resource
def get_model():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


model = get_model()


# -------------------- Schema --------------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(
    pydantic_object=Movie
)


# -------------------- Prompt --------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Extract movie information from the paragraph.

{format_instructions}

Return only the structured JSON output.
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])

# -------------------- UI --------------------

st.set_page_config(
    page_title="🎬 Movie Info Extractor",
    layout="centered"
)

st.title("🎬 Movie Information Extractor")

st.write(
    "Paste any movie description and AI will convert it into structured data."
)


paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=200
)


if st.button("Extract Data"):

    if not paragraph.strip():

        st.warning("Please enter a paragraph first.")

    else:          

        with st.spinner("Analyzing movie..."):

            try:

                # Create prompt
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions":
                        parser.get_format_instructions()
                })

                # Call Groq
                response = model.invoke(final_prompt)

                # Raw output
                st.subheader("Raw Model Output")

                st.code(
                    response.content,
                    language="json"
                )

                # Parse response
                movie_data = parser.parse(
                    response.content
                )

                # Structured output
                st.subheader("Structured Output")

                st.json(
                    movie_data.model_dump()
                )

                st.success(
                    "Extraction Completed Successfully!"
                )

            except Exception as e:

                st.error(
                    "Failed to parse response. "
                    "Model did not follow schema."
                )

                st.exception(e)
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Groq model
model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# -------------------- Schema --------------------

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


# Pydantic parser
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
"""
    ),
    (
        "human",
        "{paragraph}"
    )
])


# -------------------- Input --------------------

para = input("Give your paragraph: ")


# -------------------- Create Prompt --------------------

final_prompt = prompt.invoke({
    "paragraph": para,
    "format_instructions": parser.get_format_instructions()
})


# -------------------- Model Call --------------------

response = model.invoke(final_prompt)


# -------------------- Parse Output --------------------

movie_data = parser.parse(response.content)


# -------------------- Output --------------------

print("\nMovie Information:")
print(movie_data)
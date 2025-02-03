from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
model=OllamaLLM(model="llama3.2")
template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)
def parse_with_ollama(dom_chunk,parse_description):
  promt=ChatPromptTemplate.from_template(template)
  chain=promt | model
  parsed_result=[]
  for i, chunk in enumerate(dom_chunk,start=1):
    response=chain.invoke({"dom_content":chunk,"parse_description":parse_description})
    print(f"parsed batch {i} of {len(dom_chunk)}")
    parsed_result.append(response)
  return "\n".join(parsed_result)


def split_dom_content(dom_content,max_length=6000):
    return [
        dom_content[i:i+max_length] for i in range(0,len(dom_content),max_length)
    ]
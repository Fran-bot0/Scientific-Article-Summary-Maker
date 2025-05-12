from datetime import datetime
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents.base import Document
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader


PROMPT_TEMPLATE = '''You are a scientific summarization assistant. Given the full text of a scientific article, write a structured summary of the paper between 1,000 and 2,000 characters in length, but do not use bullet points or lists. The summary should be written clearly, integrating all sections into one cohesive response.
                    The summary should include the following:
                        Introduction: Briefly explain the background and motivation of the study, if provided.
                        Methods: Summarize what the researchers did — including any experimental methods, models, tools, or materials used, if stated.
                        Results: Highlight the main findings of the study, including significant data points or trends, if available.
                        Conclusions: Discuss the main takeaways, implications of the results, and any limitations or future directions mentioned by the authors.

                    Important Constraints:
                        Do not include author contributions, funding statements, acknowledgments, competing interests, or data availability sections.
                        Only summarize the main scientific content of the article (introduction through conclusions).
                        Only include information explicitly mentioned in the article. Do not assume or infer anything not directly stated.
                        If a section lacks enough detail, note that it was unclear or not covered, but do not speculate.
                        Use concise, clear language in your own words. Do not copy directly from the article. Accuracy is more important than completeness.
                        The full article should be summarized but the most important part is the conclusion.
                    Final check: Before finalizing the summary, verify that every item you mention can be clearly found in the original text.
                    
                    Context:\n{context}'''

MODEL_NAME = 'phi4'


def load_file(file_name: str='pdf.pdf') -> list[Document]:
    '''Loads the pdf file.'''
    loader = PyPDFLoader(f'files/{file_name}')
    
    return loader.load()


def is_mostly_references(page_text: str, threshold: int=35) -> bool:
    '''Checks if the pdf page is mostly composed of references.'''
    
    bracketed_refs = len(re.findall(r'\[\d+\]', page_text))                             # [1], [23]
    parens_refs = len(re.findall(r'\(\d+\)', page_text))                                # (2), (37)
    numbered_refs = len(re.findall(r'^\s*\d+\.\s', page_text, flags=re.MULTILINE))      # 3. , 65. 
    years = len(re.findall(r'\b(19|20)\d{2}\b', page_text))                             # 1999, 2023

    score = bracketed_refs + numbered_refs + parens_refs + years
    
    return score >= threshold


def remove_irrelevant_text(pages: list[Document]) -> list[Document]:
    '''Removes all page content after certain key-words like: "References", "Aknowledgments", etc...'''
    
    for i, page in enumerate(pages, start=1):
        if is_mostly_references(page.page_content):
            pages = pages[:i]
            break

    match = re.search(regex_exp, pages[-2].page_content, re.IGNORECASE | re.MULTILINE)

    if match:
        pages[-2].page_content = pages[-2].page_content[:match.start()]
        pages.pop(-1)
    else:
        match = re.search(regex_exp, pages[-1].page_content, re.IGNORECASE | re.MULTILINE)
        pages[-1].page_content = pages[-1].page_content[:match.start()]
    
    return pages


def safe_return(obj: dict, field: str) -> str:
    '''Checks if a filed of the metadata exists and, if it does, returns it, otherwise it returns "NA".'''
    
    try:
        return obj[field]
    except KeyError:
        return 'NA'


def get_metadata(data: dict[str]) -> dict[str]:
    '''Returns the relevant metadata fields in a dictionary.'''

    return {'title': safe_return(data, 'title'), 
            'year': safe_return(data, 'creationdate')[:4], 
            'author': safe_return(data, 'author'), 
            'key_words': safe_return(data, 'keywords')}


def make_summary(doc: list[Document]) -> str:
    '''LLM makes the summary with the provided relevant information.'''
    
    document_text = ' '.join([page.page_content.replace('\n', '') for page in doc])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = OllamaLLM(model=MODEL_NAME, temperature=0)
    prompt = prompt_template.format(context=document_text)

    return model.invoke(prompt)


def save_as_markdown(metadata: dict[str], summary: str, file_name: str) -> None:
    '''Saves the summary as markdown file.'''

    with open(f'output/{file_name[:-4]}_summary.md', 'w', encoding='utf‑8') as f:
        f.write(f'# Title: {metadata['title'].title()}\n\n')
        f.write(f'- **Authors:** {metadata['author']}\n')
        f.write(f'- **Year Published:** {metadata['year']}\n')
        f.write(f'- **Date Summary:** {datetime.today().date()}\n')
        f.write(f'\n## Summary:\n')
        f.write(f'{summary}')


regex_exp = r"""^\s*[-*•]?\s*\b(Acknowledgments|Reporting Summary|Data Availability|Data Availability Statement|Supplementary Information|Suplemental Information|Supplementary Material|Supporting Information|Associated Content|Supplementary Materials|Author Contributions|Authors’ contributions|Credit Authorship Contribution Statement|Contribution Statement|Author Information|Declaration of Competing Interest|Competing Interests|Conflicts of Interest|Conflict of Interest|Conflict of Interest Statement|Funding|References|Bibliography|Works Cited|Literature Cited|Orcid|Acknowledgment|Acknowledgements|Ethics Statement|Ethical Approval|Institutional Review Board Statement|Informed Consent|Notes|Endnotes|Footnotes|Corresponding Author|Author Notes|Appendix|Appendices|Preprint|Version History)\b.*"""


if __name__ == '__main__':
    file_name = input('What is the name of the file you wish to summary?\n')
    document = load_file(file_name=file_name)
    document = remove_irrelevant_text(document)
    meta = get_metadata(document[0].metadata)
    ai_summary = make_summary(doc=document)
    save_as_markdown(metadata=meta, summary=ai_summary, file_name=file_name)

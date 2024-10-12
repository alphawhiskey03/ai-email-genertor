import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

os.getenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(model="llama3-8b-8192",groq_api_key='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',temperature=0)
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
                ###Scraped text from the website
                {page_data}
                ### instruction
                your job is to extract the job posting and return them in jSON format containing following keys: 'role', 'experience', 'skills' and 'description'.
                Only return valid json
                ### VALID JSON (NO PREAMBLE)
            """
            )
        
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data", cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big, Unable to parse jobs")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self, job, links):
        email_prompt= PromptTemplate.from_template(
                        """
                        ### JOB DESCRIPTION
                        {job_description}
                        ### INSTRUCTION
                        You are sussie, a business development executive at Randstad. Randstad is an AI & Software Consulting company dedicated to facilitating
                            the seamless integration of business processes through automated tools. 
                            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
                            process optimization, cost reduction, and heightened overall efficiency. 
                            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
                            in fulfilling their needs.
                            Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
                            Remember you are Mohan, BDE at AtliQ. 
                            Do not provide a preamble.
                            ### EMAIL (NO PREAMBLE):
                        """
        )
        chain_email = email_prompt | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


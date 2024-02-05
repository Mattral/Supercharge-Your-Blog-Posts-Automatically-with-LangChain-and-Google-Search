# Supercharge-Your-Blog-Posts-Automatically-with-LangChain-and-Google-Search

Introduction
These days, artificial intelligence is changing the copywriting field by serving as a writing assistant. These language models can find spelling or grammatical errors, change tones, summarize, or even extend the content. However, there are times when the model may not have the specialized knowledge in a particular field to provide expert-level suggestions for extending parts of an article.

In this Repo, we will take you step by step through the process of building an application that can effortlessly expand text sections. The process begins by asking an LLM (ChatGPT) to generate a few search queries based on the text at hand. These queries are then will be used to search the Internet using Google Search API that, captures relevant information on the subject. Lastly, the most relevant results will be presented as context to the model to suggest better content.

We've got three variables here that hold an article's title and content (text_all). (From Artificial Intelligence News) Also, the text_to_change variable specifies which part of the text we want to expand upon. These constants are mentioned as a reference and will remain unchanged throughout the repo.

```
title = "OpenAI CEO: AI regulation ‘is essential’"

text_all = """ Altman highlighted the potential benefits of AI technologies like ChatGPT and Dall-E 2 to help address significant challenges such as climate change and cancer, but he also stressed the need to mitigate the risks associated with increasingly powerful AI models. Altman proposed that governments consider implementing licensing and testing requirements for AI models that surpass a certain threshold of capabilities. He highlighted OpenAI’s commitment to safety and extensive testing before releasing any new systems, emphasising the company’s belief that ensuring the safety of AI is crucial. Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology. Blumenthal raised concerns about various risks associated with AI, including deepfakes, weaponised disinformation, discrimination, harassment, and impersonation fraud. He also emphasised the potential displacement of workers in the face of a new industrial revolution driven by AI."""

text_to_change = """ Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology."""
```

The following diagram explains the workflow of this project.

<img align="center" src="spcyllm.avif" alt="banner">

First we generate candidate search queries from the selected paragraph that we want to expand. The queries are then used to extract relevant documents using a search engine (e.g. Bing or Google Search), which are the split into small chunks. We then compute embeddings of these chunks and save chunks and embeddings in a Deep Lake dataset. Last, the most similar chunks to the paragraph that we want to expand are retrieved from Deep Lake, and used in a prompt to expand the paragraph with further knowledge.

Remember to install the required packages with the following command: pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken. Additionally, install the newspaper3k package with version 0.2.8.

```
!pip install -q newspaper3k==0.2.8 python-dotenv
```

## Generate Search Queries
The code below uses OpenAI’s ChatGPT model to process an article and suggest three relevant search phrases. We define a prompt that asks the model to suggest Google search queries that could be used to with finding more information about the subject. The LLMChain ties the ChatOpenAI model and ChatPromptTemplate together to create the chain to communicate with the model. Lastly, it splits the response by newline and removes the first characters to extract the data. The mentioned format works because we asked the API to generate each query in a new line that starts with -. (It is possible to achieve the same effect by using the OutputParser class) Prior to running the code provided below, make sure to store your OpenAI key in the OPENAI_API_KEY environment variable.

```
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

template = """ You are an exceptional copywriter and content creator.

You're reading an article with the following title:
----------------
{title}
----------------

You've just read the following piece of text from that article.
----------------
{text_all}
----------------

Inside that text, there's the following TEXT TO CONSIDER that you want to enrich with new details.
----------------
{text_to_change}
----------------

What are some simple and high-level Google queries that you'd do to search for more info to add to that paragraph?
Write 3 queries as a bullet point list, prepending each line with -.
"""

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=template,
        input_variables=["text_to_change", "text_all", "title"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)

response = chain.run({
    "text_to_change": text_to_change,
    "text_all": text_all,
    "title": title
})

queries = [line[2:] for line in response.split("\n")]
print(queries)
```

```
['AI technology implications for elections', 'AI technology implications for jobs', 'AI technology implications for security]
```

The queries you receive from the model might differ from the results above. It is because we set the model’s temperature argument to 0.9 which makes it highly creative, so it generates more diverse results.

## Get Search Results
We must set up the API Key and a custom search engine to be able to use Google search API. To get the key, head to the [Google Cloud console](https://console.cloud.google.com/apis/credentials) and generate the key by pressing the CREATE CREDENTIALS buttons from the top and choosing API KEY. Then, head to the [Programmable Search Engine](https://programmablesearchengine.google.com/controlpanel/create) dashboard and remember to select the “Search the entire web” option. The Search engine ID will be visible in the details. You might also need to enable the “Custom Search API” service under the Enable APIs and services. (You will receive the instruction from API if required) We can now configure the environment variables GOOGLE_CSE_ID and GOOGLE_API_KEY, allowing the Google wrapper to connect with the API.

The next step is to use the generated queries from the previous section to get a number of sources from Google searches. The LangChain library provides the GoogleSearchAPIWrapper utility that takes care of receiving search results and makes a function to run it top_n_results. Then, the Tool class will create a wrapper around the said function to make it compatible with agents and help them to interact with the outside world. We only ask for the top 5 results and concatenate the results for each query in the all_results variable.

```

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# Remember to set the "GOOGLE_CSE_ID" and "GOOGLE_API_KEY" environment variable.
search = GoogleSearchAPIWrapper()
TOP_N_RESULTS = 5

def top_n_results(query):
    return search.results(query, TOP_N_RESULTS)

tool = Tool(
    name = "Google Search",
    description="Search Google for recent results.",
    func=top_n_results
)

all_results = []

for query in queries:
    results = tool.run(query)
    all_results += results
```

The all_results variable holds 15 web addresses. (3 queries from ChatGPT x 5 top Google search results) However, it is not optimal flow to use all the contents as a context in our application. There are technical, financial, and contextual considerations to keep in mind.

Firstly, the input length of the LLMs is restricted to a range of 2K to 4K tokens, which varies based on the model we choose. Although we can overcome this limitation by opting for a different chain type, it is more efficient and tends to yield superior outcomes when we adhere to the model's window size. Secondly, it's important to note that increasing the number of words we provide to the API results in a higher cost. While dividing a prompt into multiple chains is possible, we should be cautious as the cost of these models is determined by the token count. And lastly, the content that the stored search results will provide is going to be close in context. So, it is a good idea to use the most relevant results.

## Find the Most Relevant Results
As mentioned before, Google Search will return the URL for each source. However, we need the content of these pages. The newspaper package can extract the contents of a web link using the .parse() method. The following code will loop through the results and attempt to extract the content.

```
import newspaper

pages_content = []

for result in all_results:
	try:
		article = newspaper.Article(result["link"])
		article.download()
		article.parse()

		if len(article.text) > 0:
			pages_content.append({ "url": result["link"], "text": article.text })
	except:
		continue

print("Number of pages: ", len(pages_content))
```

```

Number of pages: 14
```

The output above shows that 14 pages were processed while we expected 15. There are specific scenarios in which the newspaper library may encounter difficulties extracting information. These include search results that lead to a PDF file or websites that restrict access to web scraping.

Now, it is crucial to split the saved contents into smaller chunks to ensure the articles do not exceed the model’s input length. The code below splits the text by either newline or spaces, depending on the situation. It makes sure that each chunk has 3000 characters with 100 overlaps between the chunks.

```
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)

docs = []
for d in pages_content:
    chunks = text_splitter.split_text(d["text"])
    for chunk in chunks:
        new_doc = Document(page_content=chunk, metadata={ "source": d["url"] })
        docs.append(new_doc)

print("Number of chunks: ", len(docs))
```


```
Number of chunks: 46

```

As you can see, 46 chunks of data are in the docs variable. It is time to find the most relevant chunks to pass them as context to the large language model. The OpenAIEmbeddings class will use OpenAI to convert the texts into vector space that holds semantics. We proceeded to embed both document chunks and the desired sentence from the main article that was chosen for expansion. The selected sentence was chosen at the beginning of this repo and represented by the text_to_change variable.

```
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

docs_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
query_embedding = embeddings.embed_query(text_to_change)
```
Finding the distance between the high-dimensionality embedding vectors is possible using the cosine similarity metric. It determines how close two points are within the vector space. Since the embeddings contain contextual information, their proximity indicates a shared meaning. So, the document with a higher similarity score can be used as the source.

We used the cosine_similarity function from the sklearn library. It calculates the distance between each chunk and the chosen sentence to return the index of the best three results.

```
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_indices(list_of_doc_vectors, query_vector, top_k):
    # convert the lists of vectors to numpy arrays
    list_of_doc_vectors = np.array(list_of_doc_vectors)
    query_vector = np.array(query_vector)

    # compute cosine similarities
    similarities = cosine_similarity(query_vector.reshape(1, -1), list_of_doc_vectors).flatten()

    # sort the vectors based on cosine similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # retrieve the top K indices from the sorted list
    top_k_indices = sorted_indices[:top_k]

    return top_k_indices

top_k = 3
best_indexes = get_top_k_indices(docs_embeddings, query_embedding, top_k)
best_k_documents = [doc for i, doc in enumerate(docs) if i in best_indexes]
```

## Extend the Sentence
We can now define the prompt using the additional information from Google search. There are six input variables in the template:

- title that holds the main article’s title;
- text_all to present the whole article we are working on;
- text_to_change is the selected part of the article that requires expansion;
- doc_1, doc_2, doc_3 to include the close Google search results as context.

The remaining part of the code should be familiar, as it follows the same structure used for generating Google queries. It defines a HumanMessage template to be compatible with the ChatGPT API, which is defined with a high-temperature value to encourage creativity. The LLMChain class will create a chain that combines the model and prompt to finish up the process by using .run() method

```
template = """You are an exceptional copywriter and content creator.

You're reading an article with the following title:
----------------
{title}
----------------

You've just read the following piece of text from that article.
----------------
{text_all}
----------------

Inside that text, there's the following TEXT TO CONSIDER that you want to enrich with new details.
----------------
{text_to_change}
----------------

Searching around the web, you've found this ADDITIONAL INFORMATION from distinct articles.
----------------
{doc_1}
----------------
{doc_2}
----------------
{doc_3}
----------------

Modify the previous TEXT TO CONSIDER by enriching it with information from the previous ADDITIONAL INFORMATION.
"""

human_message_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=template,
        input_variables=["text_to_change", "text_all", "title", "doc_1", "doc_2", "doc_3"],
    )
)
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)

response = chain.run({
    "text_to_change": text_to_change,
    "text_all": text_all,
    "title": title,
    "doc_1": best_k_documents[0].page_content,
    "doc_2": best_k_documents[1].page_content,
    "doc_3": best_k_documents[2].page_content
})

print("Text to Change: ", text_to_change)
print("Expanded Variation:", response)
```


```output
Text to Change:
Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications
for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating
the potential of the technology.

Expanded Variation:
During a Senate Judiciary Subcommittee on Privacy, Technology, and the Law hearing titled “Oversight of AI: Rules for Artificial Intelligence,”
Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and its implications for elections, jobs,
and security. Blumenthal even demonstrated the potential of AI voice cloning software trained on his speeches, highlighting the need for AI
regulations. Recent advances in generative AI tools can create hyper-realistic images, videos, and audio in seconds, making it easy to spread fake and
digitally created content that could potentially mislead voters and undermine elections. Legislation has been introduced that would require candidates
to label campaign advertisements created with AI and add a watermark indicating the fact for synthetic images. Blumenthal raised concerns about
various risks associated with AI, including deepfakes, weaponized disinformation, discrimination, harassment, and impersonation fraud. The Senate
Judiciary Subcommittee on Privacy, Technology, and the Law has jurisdiction over legal issues pertaining to technology and social media platforms,
including online privacy and civil rights, as well as the impact of new or emerging technologies.
```


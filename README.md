# Supercharge-Your-Blog-Posts-Automatically-with-LangChain-and-Google-Search

Introduction
These days, artificial intelligence is changing the copywriting field by serving as a writing assistant. These language models can find spelling or grammatical errors, change tones, summarize, or even extend the content. However, there are times when the model may not have the specialized knowledge in a particular field to provide expert-level suggestions for extending parts of an article.

In this Repo, we will take you step by step through the process of building an application that can effortlessly expand text sections. The process begins by asking an LLM (ChatGPT) to generate a few search queries based on the text at hand. These queries are then will be used to search the Internet using Google Search API that, captures relevant information on the subject. Lastly, the most relevant results will be presented as context to the model to suggest better content.

We've got three variables here that hold an article's title and content (text_all). (From Artificial Intelligence News) Also, the text_to_change variable specifies which part of the text we want to expand upon. These constants are mentioned as a reference and will remain unchanged throughout the lesson.

```
title = "OpenAI CEO: AI regulation ‘is essential’"

text_all = """ Altman highlighted the potential benefits of AI technologies like ChatGPT and Dall-E 2 to help address significant challenges such as climate change and cancer, but he also stressed the need to mitigate the risks associated with increasingly powerful AI models. Altman proposed that governments consider implementing licensing and testing requirements for AI models that surpass a certain threshold of capabilities. He highlighted OpenAI’s commitment to safety and extensive testing before releasing any new systems, emphasising the company’s belief that ensuring the safety of AI is crucial. Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology. Blumenthal raised concerns about various risks associated with AI, including deepfakes, weaponised disinformation, discrimination, harassment, and impersonation fraud. He also emphasised the potential displacement of workers in the face of a new industrial revolution driven by AI."""

text_to_change = """ Senators Josh Hawley and Richard Blumenthal expressed their recognition of the transformative nature of AI and the need to understand its implications for elections, jobs, and security. Blumenthal played an audio introduction using an AI voice cloning software trained on his speeches, demonstrating the potential of the technology."""
```

The following diagram explains the workflow of this project.

<img align="center" src="spcyllm.avif" alt="banner">

First we generate candidate search queries from the selected paragraph that we want to expand. The queries are then used to extract relevant documents using a search engine (e.g. Bing or Google Search), which are the split into small chunks. We then compute embeddings of these chunks and save chunks and embeddings in a Deep Lake dataset. Last, the most similar chunks to the paragraph that we want to expand are retrieved from Deep Lake, and used in a prompt to expand the paragraph with further knowledge.

Remember to install the required packages with the following command: pip install langchain==0.0.208 deeplake openai==0.27.8 tiktoken. Refer to the course introduction if you are looking for the specific versions we used to write the codes in this lesson. Additionally, install the newspaper3k package with version 0.2.8.

```
!pip install -q newspaper3k==0.2.8 python-dotenv
```


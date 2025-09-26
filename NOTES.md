# Managing and evaluating generative AI systems

Date: 25 Sept 2025

## GenAIOps vs MLOps

### Traditional MLOps Learning Lifecycle

1. Business problem
2. Goal definition: goal to solve the problem but also goal in terms of latency, etc. So you can monitor if it's not detrimental for the UX for example. You define the baseline of the model: the model will be good at these numbers.
3. Data collection and prep: takes 80% of the work! very important to test the data and understand how many data points you need to build a good model (you need many data, otherwise GIGO!) You need to have a proper data lake with data monitoring in place before building the model, because if you have bad data, you won't have a good model. Also beware of GDPR when choosing the data, no sensitive data into the model!
4. Feature engineering
5. Model training
6. Model evaluation: if the expected range is not met (i.e. the model is overfitting or underfitting), we go back to step (3) and loop until we pass evaluation
7. Model registration: where you document the model, so that it is reproducible!
8. Model deployment and serving
9. Model monitoring: monitor that there is no data drift, that the model is still working at the right range that we set in our goal definition (baseline). Very crucial to ensure that the new trends are capture and see if new data make the model drift or continue to product the expected results within our acceptable margin (based on our goals).
10. Model improvement --> back to step (3) and loop: this means that a model needs to be scalable because a model needs to be retrained continuously, so you need to feed in new data continuously --> it has to be scalable!

### LLMLOps Lifecycle

1. Business problem
2. Goal definition
3. Data collection
4. Feature engineering done by the big players (Mistral, OpenAI, Anthropic, etc.): We are not building models anymore, we use existing models
5. Optional: Fine-tuning of existing models (open source only)
6. Deployment of models (as is or fine-tuned)
7. Use the model directly with prompt engineering, or use agents, or RAG
8. Evaluation: very important because we need to be sure that things happen how we defined them in our goals

### Foundational models

The future is larger models (more and more billions of parameters; e.g. from GPT-2 to GPT-3 we went from 1B of parameters to 100B of parameters) but also smaller models that are more specialized for specific industries and produce better results because their data for training is more focused. They will use smaller models from big players and fine tune them; not build new foundational models from scratch because too costly (GPUs, etc.; cost of training GPT-3.5: 20M USD), very heavy work and you don't even know if it's going to work at the end.

So alternatives are:

- Fine-Tuning
- Prompt Engineering
- RAG

### Fine-Tuning

2 ways to do it:

1. Parameter-Efficient Fine-Tuning (PEFT): Freeze the pretrained transformer (with specific weight, etc.) and add new layers on top (easiest); update only parts of the model
2. Full Fine-Tuning: Retrain the pretrained transformer by adding new layers and revisit then all parameters

Fine-tuning continues the training of the model on a smaller, narrower dataset.

**Fine-tuning vs. RAG:**
With fine-tuning the context is built into the model vs. RAG: add context on top of the model.
With fine-tuning once the data is in there, you can't change it vs. RAG: you can change data anytime because you can always choose which context you pass.

To really fine-tune a big model we would need GPU, etc. We can do fine-tuning on our own computer but only with small models. E.g.:

- Fine-tuning LLaMA 7B parameters with 1GPU and 25M total tokens (size of dataset x nbr of times we run it (epochs)): it will take over 5h to fine-tune and cost 15USD
- Fine-tuning GPT3 175B parameters with 32GPU (to be faster) and 500M tokens: it will take 2600h to fine-tune and cost 200k USD!

Fine-tuning is good to use when the data we want to add are not going to evolve, e.g. legal data, medical, etc.

Always start small and specific, then evaluate, then move up. Never start with a big model!

!!! In real case scenario, you can do fine-tuning directly in Amazon Bedrock.

### Deployment strategies

Deployment can be:

- serverless (fully managed)
- dedicated instance on cloud (e.g. dedicated instance of OpenAI models on Azure cloud -> so the data you send to these models is not going to OpenAI, only to your instance of chatGPT)
- self-hosted

Best approach for cost and easy: serverless
Best approach for privacy sensitive: dedicated instance

**Scaling**

- Caching: Improve scaling with caching (e.g. Redis, Memcached, vector DBs).
- Batch inferencing: Scaling can be improved as well with batch processing --> this is good when latency is not key, don't have to do it live.
- Even with Live inference we can have batch techniques:
  -- fixed-size batching (process nbr of requests together with max, e.g. N = 32)
  -- time-window batching (wait for a fixed time interval and process together, e.g. T = 50ms)
  -- dynamic batching (group requests together wihin a max tokens number, e.g. 2048 tokens)

### LLM Serving

You need to have a Gateway / proxy layer between your user and the models.

Gateway with:

- authentication
- rate limiting
- guardrails (including exclusion of sensitive data, including content moderation, have clear guardrails against nudity, self harm, etc.)
- cost
- monitoring

This Gateway can route the traffic to the proper model.

Read this article: GenAI proxy at Expedia: https://medium.com/expedia-group-tech/gateways-guardrails-and-genai-models-aa606379164d
Now we have LiteLLM (https://www.litellm.ai/#features) that provides this kind of service or https://www.guardrailsai.com/

## Prompt Engineering and Management

70% of the problems can be solved by prompt engineering!

Also GIGO with prompts!

**Chain-of-thought prompting:** e.g. doctor diagnosis --> similar to fundraising diagnosis? give the info (anamnèse) then ask the LLM to do the diagnosis step by step.

Do three main instructions, then fine tune the prompt based on the output we get, section by section. We do it in an iterative way instead of putting too much info that will clutter the context window!

**Template prompting:** e.g. sentiment analysis
Check the prompt builders in chatGPT (custom GPTs)

Always start with the simplest:

1. Prompt engineering --> if does not give proper output, move to (2)
2. Fine-tuning --> if does not give proper output and if you have the resources!, move to (3)
3. Pre-training (creating your own model)

#### Prompt Management

1. Prompt repository
2. Prompt version control
3. Prompt evaluation
4. Prompt collaboration
5. LLM integration

We treat the prompts as code!

Tools for version control: _Langfuse_ (https://langfuse.com/), _Braintrust_ (https://www.braintrust.dev/), _Langsmith_, _PromptLayer_, GitHub repo.
Braintrust includes evals as well right in it.

Prompt templates, memory and chains: LangChain.

### Ollama

Enables you to download distillations of open source models to run them directly on your machine.
Only small models, so you wouldn't use Ollama for production (because it's local AND it's small models, max. 8B parameters to run on our machine; very good gamine computer can run 32B parameters), but good to run tests.

---

## RAG

### Embeddings

Check paper from Expedia: "Hotel2Vec" --> how to vectorize an hotel (complex, many data points): https://arxiv.org/abs/1910.03943

Compare the similarity between 2 vectors (= numbers) by calculating the cosine between them (the angle).

Embedding models: use the one from OpenAI because a good embedding model is an ML model that captures very well the different parameters of the content you pass to it. E.g. hotel: many criteria enable to compare one hotel from another, so we need to capture all these many parameters.
If the embedding is not good enough because the ML is not trained on specific data enough, you can also fine-tune the embedding model with your data, so then you use this fine-tuned embedding model to create your embeddings.

### Retrieval

We have to use the same model to convert the user query into an embedding, so that then you can compare in your vectorDB.
If you use another model to convert the user query, then you can't compare two different types of embedddings; they would be in a different dimension.

Try different vectorDBs and compare the latency, it can be up to 50x faster!

To be sure that the LLM answers only based on the data we have in the vectorDB, you can say it in the prompt: "answer only based on the provided context; if the answer is not part of the context, say that you don't know".

Reranking: when you retrive, you get the 10 best hotels based on similarity; then you may want to rerank this small DB to have from these 10 hotels, rank them by biggest rooms available.

### Best practices for RAG

1. Chunk your documents carefully, in small bits (approx. 100-500 words with overlap; do not chunk in pages, otherwise it's not fine enough) and use a strong encoder
2. Hybrid search: combine dense and sparse retrieval (BM25 + embeddings) to improve recall
3. Metadata filtering: tag and filter documents using metadata to improve relevance
4. Latency optimization: use async pipelines, approximate nearest neighbor (ANN) search, and cache frequent queries
5. Evaluation: monitor retrieval precision/recall, output accuracy, and latency

--> TODO: Try with our use case with annual reports: chunk by page vs. chunk in small bits (100-500 words), and compare the results.

### RAG types

1. Standard RAG: retriever --> retrived documents
2. Graph RAG: retrieve entities and relations
3. Hierarchical RAG: multi-stage retrival, e.g. coarse retriever --> fine retriever. E.g. retrive general data about Unige (top level RAG) or fine info about a specific departement (fine RAG)
4. Agentic RAG: agent planner --> task-specificc retrievers
5. Conversation RAG: retrieve user conversation history to keep context, optimized for multi-turn interactions with context windows and memory (session memory, long term memory, etc.)

### What to use when?

1. Prompt engineering (pass examples, Q&A into the context directly as text)
2. RAG
3. Fine-tuning (if data is not changing, e.g. changes once a year, could do once a year, but cost, time and need labelled data)
4. Pre-training

---

## Agents

What are they? Basically they are a workflow with an LLM on top.

Agent has:

1. Tools
2. Planning: e.g. reflection, self-critics, chain of thoughts, subgoal decomposition, etc.
3. Memory
4. Actions

Use both LangChain and Llamaindex:

- LangChain for orchestration, memory, decision-making and tool calling
- Llamaindex for access to structured and unstructured external knowledge

Check Google Mariner: https://labs.google.com/mariner/landing
The future of human-agent interaction.
or: Comet Browser by Perplexity: https://www.perplexity.ai/comet

---

## Observability, Monitoring, Evaluation and Feedback Loop

Track everything with traces by using @observe from Langfuse (e.g. 04-Agents-exercise)!
Need to collect as many data as possible (track prompts, user inputs, outputs, latency, cost, tool use, etc.).
Then we can make sense of the data through a dashboard that summarizes the traces.

Every 6 months, go back to the raw data (traces) to see if the dashboards are still appropriate, are they missing something?

Langfuse: use LLM-as-a-Judge

---

## Responsible AI

AI should be:

1. Beneficial (there must be a good reason to use AI, not only to use AI per se)
2. Fair
3. Transparent: we have to say what has been AI generated
4. Accountable: we as creator of a service using AI, we are accountable for it (i.e. what the AI creates for the user) (e.g. LLM offering a car for 10k usd to a customer instead of the normal price of 40k usd --> the customer won in court, the company could not cancel this order as they are responsible to have the LLM say the right thing and do the right thing)
5. Privacy conscious

Check the model licenses to see that they are not using our data to train the model, etc.

---

## Data Privacy and Handling Personally Identifiable Information (PII)

1. PII redaction: use automated PII detection tools (Presidio https://microsoft.github.io/presidio/, AWS Macie) before sending data to LLMs!
2. Secure API usage: TLS, OAuth, API keys
3. Data minimization: follow "least privilege" and "minimum necessary" principle (droit d'en connaître)

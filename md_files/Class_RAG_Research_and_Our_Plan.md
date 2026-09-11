# Understanding Class-RAG and Our Research Plan

A beginner-friendly guide for our one-month research assignment

**Selected paper:** [Class-RAG: Content Moderation with Retrieval Augmented Generation, version 1](https://arxiv.org/html/2410.14881v1)

**Working project title:** Relevance-Gated Retrieval for Efficient Content Moderation with Small Language Models

**Available resources:** Your reported 30 hours of Kaggle GPU access. The actual GPU model and memory still need confirmation.

**Project status:** We have selected the paper and proposed an experiment. We have not trained a model, run the experiments, or obtained results yet.

---

## 1. What are we trying to do?

We want to build a program that reads a text prompt and decides whether it is safe or unsafe under a specified content policy.

A **prompt** is the text someone gives to an AI system. A **policy** is the set of rules defining what is allowed. A **label** is the recorded answer, such as `safe` or `unsafe`.

Imagine a website that accepts descriptions for creating pictures. Before using a description, the website checks whether it is allowed. Our research concerns that checking stage. We do not need to generate pictures for this project.

The difficult part is understanding meaning. A word associated with harmful content can appear in an educational, preventive, or otherwise permitted context. Conversely, harmful meaning can be expressed without an obvious warning word.

Our labels must follow the dataset's definitions. We should not silently replace those definitions with our personal interpretation of safety.

**Our goal is to investigate a small improvement in the classification process and measure what actually changes.**

## 2. The basic ideas, explained simply

### 2.1 What is an LLM?

An LLM, or large language model, is a model trained on text that can process language and produce text responses.

For our task, we ask it to produce a restricted answer: `safe` or `unsafe`. This is called **classification** because the answer belongs to a predefined category.

The model can still misunderstand the input, misunderstand the policy, or output an unexpected answer. We must measure these failures rather than assume that fluent language means correct judgment.

### 2.2 What is retrieval?

Retrieval means searching a collection and selecting useful items.

Think of a student looking through solved examples before answering a new question. The student does not read every example. They search for examples that seem relevant.

For our project, the collection contains text examples and their labels. It is not necessary to search the live internet for every input.

### 2.3 What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

In ordinary language: search for useful information, give that information to the model, and ask the model to answer using it.

In a question-answering application, the retrieved information might be paragraphs from documents. In a classification application, it can be previously labeled examples. The generated answer can be only a class label.

### 2.4 What is an embedding?

An embedding represents text as a list of numbers. An embedding model learns representations that can be used to compare pieces of text.

For example, “a child riding a bicycle” and “a kid cycling” may have similar representations even though their words differ.

However, two texts can discuss the same topic and have different intentions. Similarity is a useful search signal, not proof of identical meaning or the same correct label.

We will not calculate these representations manually. A pretrained embedding model will produce them.

### 2.5 What is fine-tuning?

Fine-tuning changes a model's learned parameters using additional training examples. It is like giving a student extra practice and updating what they have learned.

Retrieval gives the model information during a particular prediction. It is like allowing the student to consult examples while taking a test.

These operations can be combined, but they are different:

| Operation | What changes? | When does it happen? |
| --- | --- | --- |
| Fine-tuning | The model's learned parameters | During training |
| Retrieval | The examples included in the current input | During prediction, and sometimes training |
| Prompt design | The instructions and formatting | When we construct the model input |

Showing examples in a prompt does not, by itself, permanently train the model.

## 3. What the original Class-RAG paper does

The following is a compact description of the authors' system, not our experimental result.

| Component | Role in the original system |
| --- | --- |
| Embedding model | DRAGON RoBERTa represents text for similarity search. |
| Reference collection | Separate safe and unsafe collections store prompts, labels, embeddings, and explanations. |
| Retriever | FAISS selects two nearby examples from each class using L2 distance. |
| Classifier | A fine-tuned Llama-3-8B receives the input and references and makes a classification. |

The authors use CoPro, generate supporting explanations with Llama-3-70B, and evaluate classification, robustness, and adaptation through reference updates. Their experiments report benefits over their selected baselines. Training used a machine with eight A100 80 GB GPUs. [Original paper, Sections 3–5](https://arxiv.org/html/2410.14881v1)

The paper discusses difficulties using safe examples effectively. It identifies multilingual evaluation and better reference collection construction as future directions. [Paper, Section 7](https://arxiv.org/html/2410.14881v2)

These findings motivate our interest; they do not establish that our proposed filter will work.

## 4. A classroom analogy for the research idea

Imagine answering a question about bicycle safety. A search system finds the nearest available examples in its collection.

One example discusses bicycle helmets. Another discusses railway station rules. Both involve safety, but one is much more useful for the current question.

Now imagine the collection is small and contains no good bicycle example. A nearest-neighbor search can still return the railway example. It is the best available match, but it may be a poor match overall.

This distinction is the motivation for our experiment:

> Being the closest available example does not necessarily make an example useful enough to include.

This is a hypothesis about how irrelevant context may affect a classifier. Some additional examples may help, some may distract, and others may have little effect. Only experiments can tell us what happens in our setup.

## 5. What will our extension change?

We propose adding a **relevance gate**: a rule that decides whether a retrieved example is similar enough to include.

The first version will be deliberately simple:

1. Retrieve two candidates from the safe collection and two from the unsafe collection.
2. Compute a similarity score for each candidate using our chosen embedding setup.
3. Keep candidates whose score meets a selected threshold.
4. Include the surviving examples in the classification prompt.
5. If none survive, use the same policy and classification instructions without reference examples.

The model still classifies every input. Removing all reference examples does **not** automatically make an input safe, and it does not mean that we discard that test item.

### 5.1 A numerical example

**These numbers are invented for teaching. They are not experimental measurements.**

Suppose the candidates have the following cosine similarity scores:

| Candidate | Stored label | Similarity | Keep when threshold is 0.70? |
| --- | --- | --- | --- |
| A | Safe | 0.88 | Yes |
| B | Safe | 0.75 | Yes |
| C | Unsafe | 0.58 | No |
| D | Unsafe | 0.42 | No |

Fixed retrieval would pass all four examples to the model. Our proposed filter would pass A and B.

However, this could remove useful evidence from one class. The model might then become more likely to predict the other class. We therefore need to inspect which labels survive and measure both false alarms and missed unsafe inputs.

**The value 0.70 is only an illustration.** Suitable thresholds depend on the embedding model and dataset. We will select the threshold using validation data.

### 5.2 Similarity is not confidence

A similarity of 0.88 does not mean “88% certain this input is safe.” It describes a relationship between two representations.

Likewise, a model writing “I am 95% confident” does not automatically give us a reliable probability.

We will keep three ideas separate: similarity between texts, a predicted class, and confidence in that prediction.

### 5.3 What makes this an extension?

The experimental change is the selection rule for reference examples. We will examine its effect on classification and computational cost.

Building a working interface alone would not answer that research question. The research contribution comes from the hypothesis, controlled comparison, results, and explanation of when the method succeeds or fails.

We have not established that relevance filtering is globally new. Before finalizing the claim, we will check related work on adaptive retrieval and selecting demonstrations for classifiers. Our project can be a useful assignment extension without claiming to invent the general idea of filtering retrieved context.

## 6. What exactly will we build?

We will build a notebook-based experimental pipeline with these parts:

| Part | Input | Output |
| --- | --- | --- |
| Data preparation | Dataset records | Checked records with stable IDs and splits |
| Reference indexing | Training examples | Searchable embeddings and reference metadata |
| Retrieval | A new prompt | Candidate reference IDs and scores |
| Selection | Retrieved candidates | Fixed or filtered reference lists |
| Classification | Policy, references, and prompt | Predicted label |
| Evaluation | Predictions and true labels | Metrics, comparisons, and error examples |

We will save intermediate outputs so a plotting change does not require generating every prediction again.

An example prediction record could look like this:

```json
{
  "example_id": "demo_001",
  "method": "relevance_gated",
  "prediction": "safe",
  "candidate_count": 4,
  "kept_reference_count": 2,
  "threshold": 0.7,
  "reference_ids": ["train_017", "train_283"]
}
```

This is an illustrative storage format. The true label is joined separately for evaluation; it must not be included in the classifier's test input.

## 7. What data will we use?

Our starting candidate is CoPro. Its creators provide the dataset in their [official LatentGuard repository](https://github.com/rt219/LatentGuard), including `dataset/CoPro_v1.0.json`.

First, we will inspect its structure, usage terms, labels, existing splits, and grouping information. Availability does not mean we have already verified every record or recreated the paper's exact preprocessing.

We will begin with a small, reproducibly selected pilot subset. A pilot is a cheap first experiment used to discover setup problems and estimate runtime. The final subset size will depend on measured speed and remaining compute.

### 7.1 Training, validation, and test sets

| Split | Simple meaning | What we may use it for |
| --- | --- | --- |
| Training | Practice material | Build the reference collection and train, if needed |
| Validation | Practice exam | Choose thresholds and other settings |
| Test | Final exam | Evaluate the frozen method |

We will respect official split definitions where available. We will also inspect near duplicates and related prompt groups so closely related items do not accidentally cross boundaries.

### 7.2 What is data leakage?

Data leakage happens when information that should be unavailable influences the answer or the experimental choices.

If a test prompt is stored in the reference collection with its correct label, retrieval might simply reveal the answer. The resulting score would exaggerate our system's ability.

Leakage also occurs if we repeatedly inspect final test results while choosing our threshold.

Our rules are:

- Build reference collections from training data only.
- Select settings using validation data.
- Keep test labels out of retrieval and model inputs.
- Keep each original test item and its perturbed versions together when analyzing uncertainty.
- Freeze the method before final testing.

## 8. Which experiments will answer our question?

A **baseline** is a reference method against which we compare a proposed change.

| Method | What it does | Why we need it |
| --- | --- | --- |
| A: No retrieval | Classify with policy instructions alone | Check whether retrieved examples help |
| B: Fixed retrieval | Always include two safe and two unsafe references | Provide the main comparison |
| C: Relevance gate | Filter B's candidates by similarity | Test our proposed change |
| D: Short fixed retrieval | Use a smaller predetermined reference count | Check whether shorter prompts alone explain an improvement |

All four use the same underlying classifier, data split, policy instructions, and output format. B, C, and D also share the same reference collection and embedding model.

We will choose the short fixed baseline's size on validation data. It should be reasonably comparable to C's typical reference count, while acknowledging that equal reference counts do not guarantee equal token counts.

### 8.1 How close will our baseline be to the paper?

Our initial pilot will use a small instruction model to establish that the complete pipeline works. We will call this a **Class-RAG-inspired baseline** and list the differences explicitly.

After the pilot, we will decide whether a limited parameter-efficient fine-tuning experiment fits the budget and assignment expectations. LoRA is one such approach: it trains a relatively small set of additional parameters rather than updating all model weights.

If we fine-tune, the resulting classifier must be shared across the fixed and gated comparisons. We cannot give only our proposed method a better model or more training and attribute the difference to retrieval filtering.

Our own comparisons will establish the effect in our setup. They will not establish that we outperform the published system under its original conditions.

### 8.2 What is an ablation?

An ablation changes or removes one component to see what it contributes.

For example, removing the gate from C returns us to B. This helps isolate the effect of the gate.

We will keep extra experiments limited. Changing the embedding model, classifier, prompt, and selection rule simultaneously would make the result difficult to explain.

## 9. How will we measure success?

We treat `unsafe` as the positive class when defining the following counts.

| Term | Meaning |
| --- | --- |
| True positive (TP) | Unsafe input correctly flagged |
| False negative (FN) | Unsafe input incorrectly allowed |
| False positive (FP) | Safe input incorrectly flagged |
| True negative (TN) | Safe input correctly allowed |

### 9.1 Unsafe recall: how many unsafe inputs did we catch?

`Unsafe recall = TP / (TP + FN)`

If 20 inputs are unsafe and we catch 18, recall is 18/20 = 90%.

### 9.2 False-positive rate: how many safe inputs did we wrongly block?

`False-positive rate = FP / (FP + TN)`

If 80 inputs are safe and we wrongly block 8, the false-positive rate is 8/80 = 10%.

### 9.3 Precision: how often was a flag correct?

`Unsafe precision = TP / (TP + FP)`

Using the same example, we flagged 18 unsafe inputs and 8 safe inputs. Precision is 18/26, approximately 69.2%.

### 9.4 Macro-F1: how well did we handle both classes?

F1 combines precision and recall. Macro-F1 computes an F1 score for each class and averages them, giving the classes equal weight.

We will use it alongside the other metrics. A single overall score can hide the fact that a method improved one class while damaging the other.

### 9.5 Cost and speed

A **token** is a unit of text processed by the model. It may be a word, part of a word, or punctuation.

We will record input tokens, reference counts, and latency. Latency means how long a prediction takes.

Fewer tokens do not automatically guarantee a proportional speed improvement. Retrieval overhead, batching, and hardware affect runtime. We will measure both classification time and total pipeline time where practical.

### 9.6 What would count as a useful result?

Our preferred outcome is fewer false alarms while maintaining unsafe recall. Another useful outcome is comparable classification quality with fewer tokens and lower measured runtime.

“Maintaining recall” needs a numerical definition. We will choose an acceptable tolerance on validation data before the final test, rather than decide after seeing test results.

We will use uncertainty estimates, such as paired bootstrap intervals: repeatedly resample the same test items for both methods and examine how much the difference varies. A tiny change on a small test set may be inconclusive.

**None of the numbers in this section are project results.** They are arithmetic examples only.

## 10. Why might the proposed method fail?

Research means testing an idea, including the possibility that it does not help.

| Possible problem | How we will investigate it |
| --- | --- |
| Similarity reflects topic more than intent | Inspect disagreements and retrieved examples |
| Filtering removes useful opposite-class evidence | Record retained labels and changes in unsafe recall |
| Too many inputs lose all references | Measure the zero-reference rate and its errors |
| A small model ignores examples | Compare no-retrieval and fixed-retrieval predictions |
| Shorter prompts explain the benefit | Compare with the short fixed baseline |
| Improvements disappear outside familiar inputs | Evaluate a held-out setting or controlled perturbations |

An honest negative result can still support a strong assignment: we can show a controlled experiment, explain the failure, and identify the conditions under which filtering is unreliable.

We will not repeatedly redesign the method around the final test set until something looks successful.

## 11. What does robustness testing mean?

Robustness means continuing to behave reasonably when the input changes in ways that should not change its label.

We can create a small set of variants involving letter case, extra spacing, or mild typing mistakes. We will manually check a sample because a transformation can accidentally change meaning or make text unreadable.

We will report clean and perturbed performance separately. We will not count variants of one prompt as fully independent evidence when estimating uncertainty.

This is a controlled stress test, not proof that the system handles every adversarial input.

## 12. Our one-month plan

| Period | Main work | What we should have at the end |
| --- | --- | --- |
| Days 1–7 | Understand the paper, confirm hardware, inspect data, check related work, run pilot | Verified splits, working predictions, runtime estimate |
| Days 8–14 | Complete fixed retrieval and implement the gate | Comparable baseline and proposed pipelines |
| Days 15–21 | Tune on validation data, run controlled comparisons, inspect errors | Selected settings and preliminary analysis |
| Days 22–30 | Freeze setup, run final evaluation, prepare report and slides | Reproducible results and a defensible presentation |

### 12.1 How we will use the GPU allowance

We will plan against the 30-hour allowance you reported. We have not verified whether that is your remaining allowance or a recurring allocation.

We will prepare data and analysis without a GPU where practical, cache embeddings and predictions, and estimate the full run cost from a pilot. A provisional target is to reserve about six hours for unexpected issues and final checks; this is a planning buffer, not a runtime guarantee.

We will checkpoint outputs after manageable batches. If a session stops, completed predictions should be reusable. We will also save notebook versions and copy important outputs somewhere durable instead of depending on a live session.

### 12.2 What happens if progress is slower than expected?

We reduce scope transparently: one classifier, one dataset, a fixed subset, and the essential comparisons. Additional fine-tuning, a second model, or a second dataset can be postponed.

We preserve the final test and adequate writing time. A smaller complete experiment is easier to assess than an unfinished collection of loosely connected experiments.

## 13. What will we do later, after the core experiment?

These are optional future directions, not commitments for the current month.

| Direction | Question it would investigate | Extra work needed |
| --- | --- | --- |
| Class-preserving filtering | Does retaining evidence from both classes prevent biased decisions? | Another selection rule and controlled comparisons |
| Better relevance scoring | Can a model that reads both texts judge usefulness better? | A reranker and a cost-benefit evaluation |
| Second small classifier | Does the effect hold beyond one model? | Additional inference or training |
| Second dataset | Does the method generalize beyond the first collection? | Policy and label alignment |
| Bangla or mixed-language inputs | Does the idea work across languages? | Suitable data, language-capable models, and reliable annotation |

We will choose future work based on the errors we observe. For example, if lost opposite-class evidence is the main problem, class-preserving filtering is a more informative next experiment than adding an unrelated feature.

Simply translating a test set would not establish that we have built a reliable Bangla moderation benchmark. Labels, meaning, and cultural context would require review.

## 14. How we will work together

I can explain concepts, help design experiments, produce notebooks, debug errors, review results, and help write the report and slides.

You will run the Kaggle experiments, share outputs, review ambiguous examples, and make sure the scope matches your teacher's expectations. Your teacher remains the formal academic supervisor and evaluator.

For each stage, we will answer four questions:

1. What are we doing?
2. Why do we need it?
3. How do we verify that it worked?
4. What do we do next?

You should be able to explain each major design decision. The notebooks will include explanations, and the report will distinguish source material, our implementation choices, and measured findings.

## 15. What will the final submission contain?

- A data preparation notebook with fixed splits and leakage checks.
- An experiment notebook with documented configuration and resumable outputs.
- An evaluation notebook with metrics, plots, uncertainty, and error analysis.
- Prediction files recording method settings and retrieved references.
- A report explaining the problem, related work, method, experiments, limitations, and future work.
- Presentation slides showing the research question and the evidence answering it.

The main result table will begin empty:

| Method | Macro-F1 | Unsafe recall | False-positive rate | Average input tokens | Total latency |
| --- | --- | --- | --- | --- | --- |
| No retrieval | Pending | Pending | Pending | Pending | Pending |
| Fixed retrieval | Pending | Pending | Pending | Pending | Pending |
| Relevance gate | Pending | Pending | Pending | Pending | Pending |
| Short fixed retrieval | Pending | Pending | Pending | Pending | Pending |

We will fill it only with measured results and state the hardware, test size, and units.

## 16. Your immediate next step

Open a Kaggle notebook with its GPU accelerator enabled and run this cell:

```python
!nvidia-smi
```

Send the output so we can confirm the GPU name and memory. We do not need to install large models just to perform this check.

After that, we will choose the initial model configuration, inspect CoPro, and build the first small pilot. You do not need to master every concept in this guide before starting; we will revisit each concept when it becomes relevant.

## 17. Quick self-check

Try explaining these answers in your own words:

1. **Why retrieve examples?** To provide potentially useful context for a decision.
2. **Why might filtering help?** Some available examples may be too weakly related to justify including them.
3. **Why might filtering hurt?** It can remove useful evidence or unbalance the retained classes.
4. **Why use validation data?** To select settings without tailoring them to the final exam.
5. **Why measure unsafe recall and false alarms separately?** A method can improve one while worsening the other.
6. **What do we know today?** The selected paper and proposed experiment; performance remains unknown.

## Sources and reading order

1. [Class-RAG, version 1 supplied for the assignment](https://arxiv.org/html/2410.14881v1): start with the abstract, then Sections 3, 4, 5, and 7.
2. [Class-RAG, version 2](https://arxiv.org/html/2410.14881v2): use for checking revisions; record the exact version cited in our report.
3. [Official LatentGuard repository and CoPro release](https://github.com/rt219/LatentGuard): our starting point for dataset inspection.

The classroom analogy, score examples, metric calculations, and research schedule in this guide are teaching examples and our proposed plan. They are not claims that the authors ran those exact experiments.

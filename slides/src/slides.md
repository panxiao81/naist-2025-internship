## インターン結果発表

潘　瀟

---

## Fine-Tuning Language Models for Infrastructure-as-Code Code Generation

---

## Background

There are many AI powered code products.

like:

- GitHub Copilot
- Cursor
- Even some LLM models which we can run locally: Qwen2.5-Coder, Deepseek-Coder, StarCoder...

--

But code that not for writing program is still a type of code.

For example: we have `Infrastructure as Code`

--

### What is Infrastructure as Code

Basically, It means that we write down our machine configurations, cloud resources, Kubernetes manifests... into code and manage them.

We often use some tools to apply them automatically.

--

### A example

Introduce `Terraform`, A tool for provision and managing infrastructure in any cloud platform.

--

And it use a special language called HCL, A.K.A Hashicorp Configuration Language. 

and it looks like this:  

--

```hcl[16-23]
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "us-west-2"
}

resource "aws_instance" "app_server" {
  ami           = "ami-830c94e3"
  instance_type = "t2.micro"

  tags = {
    Name = "ExampleAppServerInstance"
  }
}
```
<!-- .elementL class="data-noescape" -->

This will ensure that there is a AWS `t2.micro` EC2 instance using AMI `ami-830c94e3` and with tags `ExampleAppServerInstance`
 
---

## So what about AI in this area?

### TL:DR: Bad

--

For example, Starcoder, A Pre-trained LLM model for code, There is no HCL code in their training dataset.

And there's a paper, provides a new tool to test how well LLMs can create code for IaC.

It shows that even GPT-4, the best model in their tested models, only gets about 19% correct.

And it much lower than solving regular code problems when using testing like HumanEval.

--

![Test result](https://cdn.bytez.com/mobilePapers/v2/neurips/97835/images/7-0.png)

> Kon, Patrick T., et al. "IaC-Eval: A Code Generation Benchmark for Cloud Infrastructure-as-Code Programs." Advances in Neural Information Processing Systems 37 (2024): 134488-134506. <!-- .element: class="r-fit-text" -->

---

## What did I do

I found that these models are not trained with these domain-specfic knowleages.

--

We can use supervised fine-tuning to improve LLMs output, but models could not learn new knowledges with this way.

To let model learning new knowledges, the fine-tuning technique that we should use is called continued pre-training.

Means that we continue training a model based on an existing model.

--

For supervised fine-tuning, we need datasets in `Alpaca style`, and it looks like this:

```json
[
    {
        "Instruction": "Task we want the model to perform.",
        "Input": "Optional, but useful, it will essentially be the user's query.",
        "Output": "The expected result of the task and the output of the model."
    }
]
```

--

But for continued pre-training, we only need raw texts:

```json
[
    {
        "text": "Pasta carbonara is a traditional Roman pasta dish......"
    }
]
```

--

Also, I use LoRA for continued pre-training.

LoRA is a training technique that lets us only train a little part of trainable parameters.

--

For datasets, I used terraform’s documentation, terraform aws provider’s documentation, and OPA’s documentation.

They all written in markdown, I just download it and write a script to convert to raw data, delete markdown elements, and segmentation, then save them in disk using hugging face library’s dataset.

After all I got 4387 rows of data.

```
Dataset({
    features: ['text'],
    num_rows: 4387
})
```

--

I think it’s still not enough for fine-tuning but enough for idea validation.

But I did not do any data cleaning work.

--

I fine-tuned model based on `Qwen2.5-1.5B-Instruct`, a LLM model that already fine-tuned for chat.

And thinks to LoRA, there are only 9.7% of parameters that will be retrained. With 1 Nvidia A100 40G, I could finish training for only around 20 minutes.

---

## Result

--

The evaluation task is:

> Create a DynamoDB Contributor Insights resource for a specific table with custom settings <!-- .element: class="r-fit-text" -->

from IaC-Eval dataset, and this is one of the simplist question in this dataset.

--

The right answer should be:

```hcl
resource "aws_dynamodb_contributor_insights" "test" {
    table_name = "ExampleTableName"
}
```

--

For prompt, I also use prompt based on IaC-Eval's work, and I simplify it to get a better result.

```json 
{"role": "system", "content": "You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL."},
{
    "role": "user",
    "content": "Create a DynamoDB Contributor Insights resource for a specific table with custom settings",
},
``` 
<!-- .elementL class="data-noescape" -->

--

### Original `Qwen2.5-1.5B-Instruct`

It trys to create a new DynamoDB table with resource name `contributor_insights`.

I tried some prompt engineering works, but seems like the model have no idea with `aws_dynamodb_contributor_insights` resource.

--

```hcl
# Create a DynamoDB table with Contributor Insights enabled

resource "aws_dynamodb_table" "contributor_insights" {
  name = "MyContributorInsightsTable"

  partition_key {
    attribute_name = "Id"
    attribute_type = "S"
  }

  sort_key {
    attribute_name = "Date"
    attribute_type = "N"
  }

  tags = {
    Environment = "Production"
    Service     = "DynamoDB"
  }
}

# Create a DynamoDB table policy for Contributor Insights

resource "aws_dynamodb_table_policy" "contributor_insights" {
  name_prefix = "${var.table_name}_Insights_"

  table = aws_dynamodb_table.contributor_insights.name

  policies = [
    {
      policy_type     = "DynamoDBContributeToInsight"
      condition_keys = ["requesting_principal", "principal"]
    },
  ]
}
```

--

### My `Qwen2.5-1.5B-Instruct-IaC`

The model is not stable at all. Since when training, `loss` was not getting down in half and the bad data, The model may got overfitted.

--

But at least this time, the model knows that there is a `aws_dynamodb_contributor_insights`, But it usually stopped generation when outputing code.

For code part, it usually just output ` ```hcl` in markdown.

For example:

```md
resource "aws_dynamodb_contributor_insights" "my_table" 
This resource will create a DynamoDB Contributor Insights resource for the table named "my_table".
You can optionally enable contributor insights for an existing table without requiring a data source reference.

hcl
```

--

I tried a few time to generate the answer, and model only gave me code for two times.

--

First answer is:

```hcl
resource "aws_dynamodb_contributor_insight" "my_table_contributor_insight" {
  table_name              = "my_table"
  table_index_name        = "my_table_index"
  name                    = "my_contributor_insight"
  query_name              = "my_query_name"
  time_series_metric_name = "my_time_series_metric_name"
  schema_version          = 1
}
```

--

and this is the second time:

To create a DynamoDB Contributor Insights resource for a specific table with custom settings, you can use the following Terraform configuration:
```hcl
resource "aws_dynamodb_contributor_insights" "example" {
  # Optional: Set this to true if you want to disable logging.
  enable_logging = false
  # Optional: Set this to your AWS Region ID (e.g., us-east-1).
  region        = "us-west-2"
  # Required: Set this to the name of your table (e.g., "my-table").
  table_name    = "my-table"
}
```

--

There are only two parameters in ‘contributor_insight’ resource which is ‘table_name’, and ‘index_name’.

So even it generates code that looks fine, but they are WRONG.

[Documents](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/dynamodb_contributor_insights)

--

I tried to use some part of documentation as a prompt, and the trained LLM could predict documentation which looks like original things, shows that the model has learned something from documentations.

The funny thing is that sometimes it even output like `Note:` or `Warning:` that looks like from somewhere of documentation.

```
-> Note: This resource is available in Terraform 0.12.6 and later.
~> Warning! This resource will replace the DynamoDB Contributor Insights for this table when you create it, destroy it, or update its settings.
```

--

Looks like the trained model lost some abaility to answer questions like a chatbot, but more like documentation.

For example, sometimes it outputs:

```
~> Important: This is a beta resource and it may change in the future.
-> Note: The AWS provider must be v2.58 or higher for this resource to function properly.
The following example shows how to use the resource to create a Contributor Insights
resource for a specific DynamoDB table named example.
```

or

```
This topic shows how to create a DynamoDB Contributor Insights resource for a specific table.
DynamoDB has built-in functionality that allows you to monitor the performance of your application and optimize it based on usage patterns. You can use this feature to generate insights about attribute selections and aggregate values in your data, such as the number of distinct users and popular item names. These insights help you understand how often certain queries are used, allowing you to make optimization decisions.
For more information, see Contributor Insights.
To create a DynamoDB Contributor Insights resource, use the aws_dynamodb_contributor_insight
resource.
```

---

## Future work

- Since it's hard to get high-quality datasets in IaC, Maybe should try some ways that not relaying on training or fine-tuning like RAG, Agents and more.
- Continue to improve datasets, add more documents and code.
- Using Coding LLMs as base model.

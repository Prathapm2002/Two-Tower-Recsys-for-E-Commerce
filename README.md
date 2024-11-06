# Two-Tower Recommender System for Product Recommendations(E-Commerce)

This project demonstrates a two-tower neural network approach for building a scalable and efficient recommender system. The two-tower model structure is ideal for handling large-scale recommendation tasks, like matching products to users or queries based on embeddings. This repository contains the code, model architecture, and data processing pipelines for a two-tower recommender system designed specifically to match products with user queries.

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
  - [Query Tower](#query-tower)
  - [Product Tower](#product-tower)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Project Overview

In modern e-commerce, matching user intent (queries) with relevant products is critical for optimizing user experience. The two-tower architecture used in this project offers a scalable solution for recommendation tasks by independently encoding user and product features, then calculating their similarity in a shared embedding space.

This approach powers product recommendation engines at scale, making it a popular choice for companies like Amazon. By leveraging separate "towers" for queries and products, the model achieves a balance of accuracy and efficiency, enabling real-time matching and retrieval for personalized recommendations.

## Architecture

The two-tower model is divided into two main components:
1. **Query Tower**: Encodes features from user queries or profile data.
2. **Product Tower**: Encodes product-specific information.

Each tower independently learns an embedding for the respective data, which are then compared in a common space to generate recommendations.

### Query Tower
The Query Tower takes user query inputs, transforms them into a dense representation through a neural network, and outputs an embedding. This process allows the model to capture nuanced aspects of user intent.

Key steps:
- **Embedding Layer**: Converts categorical data into dense vectors.
- **Dense Layers**: Series of fully connected layers to learn higher-order representations.
- **Normalization and Dropout**: Regularization steps to prevent overfitting and improve generalization.

### Product Tower
The Product Tower performs a similar operation as the Query Tower but focuses on product data. This tower generates an embedding for each product based on its attributes, allowing the system to match products with user queries in the embedding space.

Key steps:
- **Embedding Layer**: Converts product attributes into dense vectors.
- **Dense Layers**: Processes the product features into a comparable embedding space.
- **Normalization and Dropout**: Ensures robust learning.

### Shared Embedding Space
After the two towers independently produce embeddings, a similarity function, typically cosine similarity or dot product, is used to measure alignment between query and product embeddings. This similarity score is used to rank and recommend products to the user.

## Data Preprocessing

Data preprocessing is crucial for handling diverse data types and ensuring optimal model performance. This includes:
- **Data Cleaning**: Removing or imputing missing values.
- **Feature Engineering**: Converting categorical features into numerical representations.
- **Normalization**: Scaling features to a consistent range, facilitating faster convergence during training.

The notebook demonstrates these preprocessing steps to ensure data is in a format suitable for the model’s embedding layers.

## Model Training and Evaluation

The training pipeline utilizes a contrastive learning approach:
1. **Positive Pairs**: Matching product-query pairs.(Based on ESCI-E(Exact))
2. **Negative Sampling**: Introducing non-matching pairs to improve model generalization.(Based on ESCI-I(irrelavance)

### Training Details
- **Optimizer**: Uses an Adam optimizer for faster convergence.
- **Loss Function**: Contrastive loss (or a similar loss metric) is applied to maximize the similarity of positive pairs while minimizing it for negative pairs.
- **Evaluation Metrics**: Includes Top-N Recommendations, val_loss, and recall to measure recommendation quality.

  
### How it Works
**User Input**: A query is entered by the user, such as "wireless headphones."
**Embedding Generation**: The input is passed through the Query Tower to generate an embedding.
**Product Embeddings**: The product catalog is passed through the Product Tower to generate embeddings for each product.
**Similarity Calculation**: Cosine similarity (or another metric) is computed between the query embedding and the product embeddings.
**Ranking**: The products are ranked based on their similarity score with the query, and the top N products are recommended.

## Usage

### Prerequisites
- Python 3.x
- Libraries: TensorFlow, Pandas, NumPy, and any other required dependencies

Install dependencies via:
```bash
pip install -r requirements.txt
```

### Running the Model
1. **Preprocess Data**: Execute the data preprocessing steps in the notebook or script.
2. **Train the Model**: Run the training cells to build the model and start training.
3. **Evaluate and Test**: Evaluate the model using test data to measure recommendation quality.

### Example Output
The model outputs a ranked list of recommended products for each user query, which can be used for further integration into applications.

## Future Work

To further improve this two-tower recommender system:
- **Hyperparameter Tuning**: Experiment with different layer sizes, learning rates, and dropout rates.
- **Additional Features**: Incorporate user demographics, product ratings, and other contextual data.
- **Fine-Tuning with Domain Data**: Customize the model with specific product or query data for enhanced relevance.

## Acknowledgments

This project draws inspiration from large-scale recommendation engines used in industry. Special thanks to the TensorFlow and Keras communities for providing resources and examples.

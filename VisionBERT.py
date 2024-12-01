import os
from typing import Dict
from datasets import Dataset
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from torch.nn import CrossEntropyLoss
import pickle

os.environ['OMP_NUM_THREADS'] = '7'


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int = None):
        """
        Custom loss computation with sample weights
        """
        labels = inputs.get("labels")
        weights = inputs.get("weight")

        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items()
                           if k not in ["weight", "labels"]})
        logits = outputs.get("logits")

        # Add labels back to outputs
        outputs["labels"] = labels

        # Compute weighted loss
        if weights is not None:
            weights = weights.to(logits.device)
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.view(-1))

            # Adjust weights if num_items_in_batch is provided
            if num_items_in_batch:
                weights = weights[:num_items_in_batch]

            loss = (loss * weights.view(-1)).mean()
        else:
            loss_fct = CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.view(-1))

        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss


def create_feature_vector(df):
    """Create numerical feature vector for clustering with sample size weighting, handling missing/unseen labels."""

    # Initialize LabelEncoders
    le_gender = LabelEncoder()
    le_risk = LabelEncoder()

    # Fit and transform while handling missing values
    gender_encoded = le_gender.fit(df['Gender'].unique()).transform(df['Gender'].fillna('Unknown'))
    risk_encoded = le_risk.fit(df['RiskFactor'].unique()).transform(df['RiskFactor'].fillna('Unknown'))

    # Create age groups numerical representation with a default for missing values
    age_map = {
        '12-17 years': 0,
        '18-39 years': 1,
        '40-64 years': 2,
        '65-79 years': 3,
        '80 years and older': 4  # Include all possible labels, even if missing
    }

    # Use `.get()` with a default value for missing/unseen age groups
    age_encoded = df['Age'].map(lambda x: age_map.get(x, -1))

    # Combine features
    features = np.column_stack([
        age_encoded,
        gender_encoded,
        risk_encoded,
        df['Sample_Size'].values  # Add sample size as a feature
    ])

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, scaler


def weighted_kmeans(X, sample_weights, n_clusters, max_iter=300, random_state=42):
    """Custom K-means implementation that considers sample weights"""
    n_samples = X.shape[0]

    # Initialize centroids randomly from the weighted distribution
    rng = np.random.RandomState(random_state)
    weighted_indices = rng.choice(n_samples, size=n_clusters, p=sample_weights / sample_weights.sum())
    centroids = X[weighted_indices]

    for _ in range(max_iter):
        # Assign points to nearest centroid
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        # Update centroids using weighted means
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                weights_k = sample_weights[mask]
                new_centroids[k] = np.average(X[mask], axis=0, weights=weights_k)

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def prepare_data(file_path='data/Vision_Survey_Cleaned.csv'):
    """Load and prepare the vision health dataset with sample-size-aware clustering."""
    print("\nLoading and preparing data...")
    df = pd.read_csv(file_path)

    # Filter data
    vision_cat = ['Best-corrected visual acuity']
    df = df[df['Question'].isin(vision_cat)].copy()
    df = df[df["RiskFactor"] != "All participants"]
    df = df[df["RiskFactorResponse"] != "Total"]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Create feature vectors for clustering
    features_scaled, scaler = create_feature_vector(df)

    # Normalize sample sizes for weights
    sample_weights = df['Sample_Size'].values
    sample_weights = sample_weights / sample_weights.sum()

    # Apply weighted clustering
    n_clusters = min(5, len(df))
    clusters, centroids = weighted_kmeans(
        features_scaled,
        sample_weights,
        n_clusters=n_clusters
    )

    # Add clusters as a column
    df['cluster'] = clusters

    # Calculate cluster importance based on total sample size in each cluster
    cluster_total_samples = df.groupby('cluster')['Sample_Size'].sum()
    cluster_weights = cluster_total_samples / cluster_total_samples.sum()

    # Enhanced feature engineering with clustering information
    df['doc'] = df.apply(
        lambda x: f"""
        Patient Demographics:
        - Age Category: {x['Age']}
        - Gender: {x['Gender']}

        Risk Factors:
        - {x['RiskFactor']}: {x['RiskFactorResponse']}

        Additional Information:
        - Sample Size: {x['Sample_Size']}
        - Cluster Profile: {x['cluster']} (Weight: {cluster_weights.get(x['cluster'], 0):.3f})
        """.strip(),
        axis=1
    )

    # Encode labels
    le = LabelEncoder()
    df['labels'] = le.fit_transform(df['Response'].astype(str))

    # Combine sample size weights with cluster importance
    df['weight'] = df.apply(
        lambda x: (x['Sample_Size'] / df['Sample_Size'].sum()) *
                  cluster_weights.get(x['cluster'], 0),
        axis=1
    )

    # Create train and test splits with stratification
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['labels'],
        random_state=42
    )

    # Convert to dict format
    train_data = {
        'doc': train_df['doc'].tolist(),
        'labels': train_df['labels'].tolist(),
        'weight': train_df['weight'].tolist()
    }

    test_data = {
        'doc': test_df['doc'].tolist(),
        'labels': test_df['labels'].tolist(),
        'weight': test_df['weight'].tolist()
    }

    # Convert to datasets
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    dataset_dict = {
        'train': train_dataset,
        'test': test_dataset
    }

    # Print detailed dataset statistics
    print("\nDataset Summary:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print("\nCluster Distribution:")
    for i in range(n_clusters):
        cluster_mask = df['cluster'] == i
        cluster_samples = df[cluster_mask]['Sample_Size'].sum()
        print(f"\nCluster {i} (Total samples: {cluster_samples:,}, Weight: {cluster_weights.get(i, 0):.3f}):")
        print("Most common characteristics:")
        for col in ['Age', 'Gender', 'RiskFactor']:
            values = df[col][cluster_mask].value_counts().head(3)
            samples = df[cluster_mask].groupby(col)['Sample_Size'].sum().sort_values(ascending=False).head(3)
            print(f"{col}:")
            for val, count in values.items():
                sample_count = samples.get(val, 0)  # Use .get() for safety
                print(f"  - {val}: {count} groups ({sample_count:,} individuals)")

    print("\nLabel Distribution:")
    for label, idx in zip(le.classes_, range(len(le.classes_))):
        count = (df['labels'] == idx).sum()
        total_size = df[df['labels'] == idx]['Sample_Size'].sum()
        print(f"{label}: {count} groups, {total_size:,} individuals")

    return dataset_dict, le



def main():
    # Setup
    output_dir = "models/vision-classifier"
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset_dict, label_encoder = prepare_data()

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Define tokenization function within main to have access to tokenizer
    def tokenize_function(examples):
        """Tokenize the input texts and maintain the correct column names"""
        tokenized = tokenizer(
            examples["doc"],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors=None
        )
        # Keep the additional columns
        tokenized['labels'] = examples['labels']
        tokenized['weight'] = examples['weight']
        return tokenized

    # Tokenize the datasets
    tokenized_datasets = {}
    for split, dataset in dataset_dict.items():
        tokenized_datasets[split] = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['doc']
        )

    # Print sample to verify
    print("\nSample tokenized data:", tokenized_datasets["train"][0])

    # Initialize the model
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_encoder.classes_),
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)},
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on device: {device}")

    # Move model to device
    model.to(device)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=7,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        push_to_hub=True,
    )

    # Create the Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()

    # Save the model
    print("\nSaving model...")
    trainer.save_model(output_dir=os.path.join(output_dir, "model"))

    # Save the tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    # Save the label encoder
    label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    return trainer, model, tokenizer, label_encoder


def evaluate_model(model, eval_dataset, tokenizer, label_encoder, device) -> Dict:
    """
    Evaluate model performance using multiple metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []

    # Process each example in evaluation dataset
    for item in eval_dataset:
        # Tokenize input
        inputs = tokenizer(
            item['doc'],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.append(item['labels'])

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average='weighted'
    )

    # Calculate per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average=None
    )

    # Create confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Combine metrics
    metrics = {
        'accuracy': accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'confusion_matrix': conf_matrix,
        'per_class_metrics': {
            label: {
                'precision': p,
                'recall': r,
                'f1': f
            } for label, p, r, f in zip(
                label_encoder.classes_,
                per_class_precision,
                per_class_recall,
                per_class_f1
            )
        }
    }

    return metrics


def print_evaluation_report(metrics: Dict, label_encoder):
    """
    Print formatted evaluation report
    """
    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)

    print("\nOverall Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 50)
    print(f"{'Class':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 50)

    for label, class_metrics in metrics['per_class_metrics'].items():
        print(
            f"{label:<30} {class_metrics['precision']:>10.4f} {class_metrics['recall']:>10.4f} {class_metrics['f1']:>10.4f}")

    print("\nConfusion Matrix:")
    print("-" * 50)
    conf_matrix = metrics['confusion_matrix']
    print(conf_matrix)


if __name__ == "__main__":
    output_dir = "models/vision-classifier"
    model_path = os.path.join(output_dir, "model")
    tokenizer_path = os.path.join(output_dir, "tokenizer")

    if os.path.exists(model_path):
        print("\nLoading pre-trained model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            label_encoder_path = os.path.join(output_dir, "label_encoder.pkl")
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
            else:
                print("Warning: Label encoder not found. Running full training...")
                trainer, model, tokenizer, label_encoder = main()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            print(f"Model loaded successfully and moved to {device}")

            # Load test dataset for evaluation
            dataset_dict, _ = prepare_data()

            # Run evaluation
            print("\nEvaluating model performance...")
            eval_metrics = evaluate_model(
                model,
                dataset_dict['test'],
                tokenizer,
                label_encoder,
                device
            )

            # Print evaluation report
            print_evaluation_report(eval_metrics, label_encoder)

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Running full training instead...")
            trainer, model, tokenizer, label_encoder = main()
    else:
        print("\nNo pre-trained model found. Running training...")
        trainer, model, tokenizer, label_encoder = main()


    def predict_vision_status(text, model, tokenizer, label_encoder):
        """Make prediction using the loaded/trained model"""
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            # Convert to numpy array
            probabilities = probabilities.cpu().numpy()[0]

            # Create list of (label, probability) tuples
            predictions = []
            for idx, prob in enumerate(probabilities):
                label = label_encoder.inverse_transform([idx])[0]
                predictions.append((label, float(prob)))

            # Sort by probability in descending order
            predictions.sort(key=lambda x: x[1], reverse=True)

        return predictions

    example_text = "Age: 40-64 years, Gender: Female, Diabetes: No"
    predictions = predict_vision_status(example_text, model, tokenizer, label_encoder)

    print(f"\nPredictions for: {example_text}")
    print("\nLabel Confidence Scores:")
    print("-" * 50)
    for label, confidence in predictions:
        print(f"{label:<30} {confidence:.2%}")

# GitHub Agent for .NET Runtime Issues and PRs

A GitHub agent that can answer and propose code changes for .NET runtime issues and PRs using a fine-tuned language model. The agent leverages context from all public .NET GitHub repositories to provide accurate and actionable responses.

## Project Structure

- `agent/`: Contains the core implementation of the GitHub agent.
- `github-crawler/`: Scripts and utilities for data collection and preprocessing, including extracting issues and PRs from GitHub.
- `model/`: Code for fine-tuning and deploying the language model, including training scripts and model configurations.
- `scripts/`: Deployment and automation.

## Goals

- Fine-tune a language model to understand and propose solutions for .NET runtime issues.
- Deploy the model as a GitHub agent on Azure.
- Implement a feedback loop for continuous improvement.

## Quick start

1. Collect raw data from Github repository
2. Generate dataset
3. Finetune model

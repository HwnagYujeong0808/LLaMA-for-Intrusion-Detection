# TinyLLaMA for Cyber GNN dataset
## Baseline

## Darknet Dataset Results

| Model Name                                   | Phase   | Training/Val/Test Set Size | Epochs | Number of Batch | Batch Size | Learning Rate | Test Loss                                | Test F1 Score                            | Model Path                                     |
|----------------------------------------------|---------|----------------------------|--------|-----------------|------------|---------------|------------------------------------------|-------------------------------------------|------------------------------------------------|
| TinyLlama (w/ edge features) - fixed seed, continuous learning | Phase 1 | 874 / 98 / 108            | 10     | 87              | 10         | 1.00E-05      | 0.5793                                   | 0.9191                                    | `model/20241109-164753_llm_w_edgefeat.pth`    |
|                                              | Phase 2 | 864 / 108 / 108            | 10     | 87              | 10         | 1.00E-05      | 0.1116<br>another unused test set: 0.2312 | 0.9907<br>another unused test set: 0.9814 |                                                |

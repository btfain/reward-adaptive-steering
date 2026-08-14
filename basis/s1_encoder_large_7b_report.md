# Dedicated-encoder router — large_7b (distilroberta-base, fine-tuned encoder)

Router = small encoder on the prompt TEXT -> swing-vector regression -> argmax. 450 valid prompts (270 train / 90 val / 90 eval), K=8 moves, same split as router_explore. Targets from swing_train.npz (m_swing); early stop on val MSE.

- **eval ΔRM +0.223 [-0.030, +0.484]**  (train +0.690, val +0.154)
- single move +0.399 [+0.180, +0.637];  naive oracle +1.108 [+0.919, +1.320]  (run's de-biased oracle ≈ +0.81)

## Reading
- **eval clears single (CI above the single point)** ⇒ a purpose-trained TEXT router extracts conditioning the LLM's own state could not ⇒ the method lives; scale + confirm on the honest test.
- **eval ≈/below single AND train≈eval** ⇒ even a dedicated encoder can't predict the best move from the prompt text ⇒ strong evidence the per-prompt choice needs trial info, not just the text.
- **train ≫ eval** ⇒ still overfitting; try --freeze, smaller encoder, or more data.

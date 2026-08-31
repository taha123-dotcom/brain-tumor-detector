# Ethical Considerations — Brain Tumor Classification System

## 1. Data Privacy & Security

### Data Handling
- All MRI images are sourced from publicly available research datasets (not collected from live patients)
- No Protected Health Information (PHI) is stored in the application database
- The prediction database stores only: anonymized file ID, predicted class, confidence score, and timestamp
- No patient names, medical record numbers, or demographic identifiers are captured or retained

### Data Protection Measures
- PostgreSQL connections use SSL/TLS encryption (`sslmode=require`)
- Database credentials are managed via environment variables (not hardcoded in production)
- Images are processed in memory and not persisted on the server after inference
- No data is shared with third parties or transmitted outside the application environment

### Compliance Awareness
- This system is designed with HIPAA-aware principles: minimum necessary data, no PHI storage, encrypted transit
- GDPR considerations: no personal data collection, right-to-delete via database record removal
- If deployed in a clinical setting, a full Data Protection Impact Assessment (DPIA) would be required

## 2. Algorithmic Bias & Fairness

### Known Bias Risks
- **Class imbalance**: The training dataset contains 2,004 glioma, 2,004 meningioma, 2,048 tumor, and only 500 healthy samples. This imbalance may bias the model toward tumor classes
- **Healthy class fallback**: The healthy class is not trained — it is handled via a probability threshold (0.30). This is a heuristic, not a learned decision, and may produce inconsistent results
- **Demographic bias**: No patient age, gender, ethnicity, or imaging设备 metadata is recorded. The model may perform differently across demographic groups, and this has not been audited

### Mitigation Strategies
- The healthy threshold (0.30) was tuned to minimize false negatives for the tumor classes
- Class weights could be applied during training to address imbalance (not yet implemented)
- Future work should include demographic-aware evaluation and fairness metrics (equalized odds, demographic parity)

### Acknowledged Limitations
- The model has not been tested on images from different MRI scanners, protocols, or populations
- Performance metrics may not generalize to underrepresented groups
- No fairness audit has been conducted across demographic subgroups

## 3. Misclassification Consequences

### Medical Risk
- **False negatives** (missed tumor) could delay life-threatening conditions — this is the highest-severity error
- **False positives** (healthy classified as tumor) could cause unnecessary patient anxiety and follow-up procedures
- The model achieves 99.34% accuracy, but even 0.66% error rate has clinical significance

### Safety Mechanisms
- The healthy fallback threshold provides a safety buffer: low-confidence predictions default to "healthy" rather than forcing a tumor classification
- All predictions include confidence scores, enabling clinicians to flag low-confidence results for manual review
- The system is explicitly **not a diagnostic tool** — it is a decision-support aid that requires professional medical interpretation

### Responsible Use Disclaimer
> **This system is intended for research and educational purposes only.**
> It must not be used as the sole basis for clinical diagnosis or treatment decisions.
> All predictions should be reviewed by qualified medical professionals.
> The developers assume no liability for clinical decisions made using this system.

## 4. Stakeholder Interests

| Stakeholder | Interest | Tension |
|---|---|---|
| **Patients** | Accurate, fast diagnosis; data privacy | Privacy vs model accuracy (more data = better models) |
| **Clinicians** | Reliable second opinion, low false-negative rate | Automation vs professional judgment |
| **Developers** | Open research, reproducibility | Transparency vs intellectual property |
| **Institutions** | Cost reduction, scalability | Speed vs thorough clinical validation |
| **Regulators** | Patient safety, compliance | Innovation speed vs safety verification |

### Balancing Act
- The system prioritizes patient safety (low false-negative threshold) over specificity
- Open-source model artifacts support reproducibility while respecting dataset licensing
- The deployment pipeline (Docker + cloud) balances accessibility with controlled access

## 5. Environmental & Resource Considerations

- Training was completed in ~17.7 minutes on standard hardware, with modest energy consumption
- The ONNX runtime enables efficient inference without GPU dependency
- The Docker image uses a slim base (python:3.11-slim) to minimize resource footprint
- No large-scale compute resources are required for deployment

## 6. Future Ethical Requirements

For clinical deployment, the following would be required:
- [ ] Institutional Review Board (IRB) approval
- [ ] FDA/CE clearance as a medical device
- [ ] Formal fairness audit across demographics
- [ ] Clinical validation trial on diverse populations
- [ ] Patient consent framework for data usage
- [ ] Incident reporting mechanism for misclassifications
- [ ] Regular model retraining with updated, audited datasets

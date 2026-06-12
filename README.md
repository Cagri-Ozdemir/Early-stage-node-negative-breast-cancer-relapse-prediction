# Early-stage-node-negative-breast-cancer-relapse-prediction
To get a new patient's probability of relapse risk and clinicopathological and genomic features contributions to this decision use **deploy final model.py**

Update the working directory path in the script to match your local setup:

```python
base_dir = '/path/to/your/working/directory'  # Replace with your local project path
```

### 🚀 Running a Prediction

To test the model with a custom case, update the example patient dictionary in the script with your own variables and execute **deploy final model.py**


```python
# Update example patient variables and run the file:
example_patient = {
    'FGA'              : 0.05,
    'TMB'              : -0.30,
    'Age'              : 0.70,
    'Grade'            : 1,
    'ER Status'        : 1,
    'PR Status'        : 1,
    'HER2 Status'      : 0,
    'Menopausal Status': 1,
    'TP53 SNVs'        : 0,
    'TP53 CADD'        : 0.0,
    'PIK3CA SNVs'      : 1,
    'PIK3CA CADD'      : 15.0,
    'CCND1 CNAs'       : 0.0,
    'MDM4 CNAs'        : 0.0,
}
```

🔵 **f(x) < 0.5: Low Risk**

🔴 **f(x) > 0.5: High Risk**

<img width="5353" height="4008" alt="example_patient_shap2" src="https://github.com/user-attachments/assets/8fa97801-81c5-48f2-9364-87110108f83b" />

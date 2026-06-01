# Spin Coating Thin-Film Simulator

Streamlit web simulator for spin coating thin-film uniformity using the Emslie-Bonner-Peck model with Meyerhofer-type viscosity growth and evaporation.

## Model Features

- Uniform film thinning: `dh/dt = -2*K(t)*h^3 - E`
- Exact EBP parameter: `K(t) = rho*omega^2/(3*eta(t))`
- Viscosity growth: `eta(t) = eta0*exp(alpha*t)`
- Flow-to-evaporation transition detection: `2*K(t)*h^3 = E`
- Gaussian radial initial profile
- Radial finite-difference thin-film evolution
- Final uniformity metric: `U = (h_max - h_min)/h_mean * 100`
- Analytical EBP validation view
- Process feasibility map over spin speed and initial viscosity

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and `README.md`.
3. Go to Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy and submit the public app URL with the report.


import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Simulate forward rates and calculate cap and floor prices
def calculate_prices(volatility, cap_rate, floor_rate, discount_rate, forward_curve, sofr_curve):
    n_steps = 100
    n_simulations = 10
    dt = 1 / 252
    maturities = np.array([1, 2, 5, 10, 30])
    simulated_forward_rates = np.zeros((n_steps, len(maturities), n_simulations))
    
    # Initialize forward rates from the last row of the forward curve
    simulated_forward_rates[0, :, :] = forward_curve.iloc[-1, :].values[:, np.newaxis]
    
    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt), (len(maturities), n_simulations))
        diffusion = volatility * dW
        simulated_forward_rates[i, :, :] = simulated_forward_rates[i-1, :, :] * np.exp(diffusion)
    
    # Calculate payoffs
    cap_payoffs = np.zeros((n_steps, n_simulations))
    floor_payoffs = np.zeros((n_steps, n_simulations))
    
    for i in range(1, n_steps):
        for j in range(n_simulations):
            for k in range(len(maturities) - 1):
                fwd_rate = simulated_forward_rates[i, k, j]
                cap_payoffs[i, j] += max(fwd_rate - cap_rate, 0)
                floor_payoffs[i, j] += max(floor_rate - fwd_rate, 0)
    
    # Use SOFR-based discount factors
    discount_factors = np.exp(-sofr_curve.iloc[-1] * np.arange(1, n_steps + 1) * dt)
    average_cap_payoff = np.mean(cap_payoffs, axis=1)
    cap_price = np.sum(average_cap_payoff * discount_factors)
    average_floor_payoff = np.mean(floor_payoffs, axis=1)
    floor_price = np.sum(average_floor_payoff * discount_factors)
    
    return cap_price, floor_price

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])

# App layout
app.layout = html.Div([
    html.H1("Interactive Sensitivity Analysis for Cap and Floor Pricing", style={"textAlign": "center"}),
    
    html.Div([
        html.Label("Volatility"),
        dcc.Slider(id="volatility-slider", min=0.01, max=0.03, step=0.001, value=0.02,
                   marks={i: f"{i:.2f}" for i in np.linspace(0.01, 0.03, 5)}),
        
        html.Label("Cap/Floor Rate"),
        dcc.Slider(id="cap-rate-slider", min=0.02, max=0.04, step=0.001, value=0.03,
                   marks={i: f"{i:.2f}" for i in np.linspace(0.02, 0.04, 5)}),
        
        html.Label("Discount Rate"),
        dcc.Slider(id="discount-rate-slider", min=0.005, max=0.02, step=0.001, value=0.01,
                   marks={i: f"{i:.3f}" for i in np.linspace(0.005, 0.02, 4)}),
    ], style={"width": "80%", "margin": "auto"}),

    dcc.Graph(id="cap-floor-pricing-graph", style={"height": "700px"})
])

# Callback to update the graph based on slider values
@app.callback(
    Output("cap-floor-pricing-graph", "figure"),
    [Input("volatility-slider", "value"),
     Input("cap-rate-slider", "value"),
     Input("discount-rate-slider", "value")]
)
def update_graph(volatility, cap_rate, discount_rate):
    floor_rate = cap_rate  # For simplicity, set floor rate equal to cap rate
    cap_price, floor_price = calculate_prices(volatility, cap_rate, floor_rate, discount_rate)
    
    # Create a bar chart for cap and floor prices
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Cap Price"], y=[cap_price], name="Cap Price", marker_color="blue"))
    fig.add_trace(go.Bar(x=["Floor Price"], y=[floor_price], name="Floor Price", marker_color="green"))
    fig.update_layout(
        title=f"Cap and Floor Prices (Volatility={volatility:.2f}, Cap/Floor Rate={cap_rate:.2f}, Discount Rate={discount_rate:.3f})",
        xaxis_title="Instrument",
        yaxis_title="Price",
        legend_title="Instrument",
        template="plotly_white"
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
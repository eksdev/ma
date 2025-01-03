import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import stripe

# Import the WS1 class for Similar Stocks
from assets.func.similar import WS1

from assets.fundamental import get_metrics
from assets.sectorseparate import sectorseparate

def convert_to_number(value):
    """
    Converts a string value with M (million) or B (billion) suffixes 
    into a numeric value.
    """
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('B'):
            return float(value[:-1]) * 1e9
        elif value.endswith('M'):
            return float(value[:-1]) * 1e6
    try:
        return float(value)
    except ValueError:
        return value  # Return original if conversion fails

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fundamental Tab Setup
FUND_COLUMNS = [
    "Ticker", "P/E", "Insider Own", "Market Cap", "Forward P/E", 
    "Income", "Sales", "ROA", "Short Interest", "ROE", 
    "Beta", "Employees", "Sales Y/Y TTM"
]
FUND_COLUMNS_DEFS = [{"name": col, "id": col} for col in FUND_COLUMNS]

# Set your Stripe secret key
stripe.api_key = "sk_test_51QbkRK4SsxaWFVRwopdUF8UxbQ9IMVUXNpKujX22w3t13Nk7zim0pMgwzCmCBkxYfSullTwqaFNMl4S5ZUgS3xpJ00IKPUYN8v"

# Path to the CSV file for sector data
file_path = 'assets/COMPREHENSIVE.csv'

# Load sector data
try:
    sector_dataframes = sectorseparate(file_path)
except Exception as e:
    print(f"Error loading sector data: {e}")
    sector_dataframes = {}

# Prepare top 5 data for specified sectors
def get_top_5_by_sector(dataframes, sectors):
    top_5_tables = {}
    for sector in sectors:
        if sector in dataframes:
            top_5_tables[sector] = dataframes[sector].head(5)
    return top_5_tables

selected_sectors = ["Consumer Defensive", "Consumer Cyclical"]
top_5_by_sector = get_top_5_by_sector(sector_dataframes, selected_sectors)

# Initialize the Dash app with suppress_callback_exceptions
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.Img(
        src='/assets/EIQ2.png',
        className="float-effect",
        style={
            'width': '200px', 
            'height': 'auto', 
            'display': 'block',
            'marginLeft': 'auto', 
            'marginRight': 'auto', 
            'padding': '1rem'
        }
    ),

    dcc.Store(id='metrics-store', data=[]),

    dcc.Tabs([
        # ---------------- Home Tab ----------------
        dcc.Tab(label="Home", children=[
            html.Div([
                html.H1(
                    "Welcome to EQuity",
                    className="fade-in-up",
                    style={"textAlign": "center", "color": "var(--color-primary)", "marginTop": "1rem"}
                ),
                html.Div([
                    html.Div([
                        html.P(
                            [
                                html.B("EQuity ", style={"fontWeight": "bold", "color": "var(--color-primary)"}),
                                "is a Quantitative Market Research Group based in Rhode Island..."
                            ],
                            className="fade-in-up",
                            style={
                                "textAlign": "justify", 
                                "marginBottom": "1rem", 
                                "fontSize": "1.2rem", 
                                "lineHeight": "1.8"
                            }
                        ),
                        html.P(
                            "Our approach is deeply rooted in advanced statistical and computational techniques...",
                            className="fade-in-up",
                            style={
                                "textAlign": "justify", 
                                "marginBottom": "1rem", 
                                "fontSize": "1.2rem", 
                                "lineHeight": "1.8"
                            }
                        ),
                        html.P(
                            "In addition to identifying undervalued assets, our methods include robust sector-specific analyses...",
                            className="fade-in-up",
                            style={
                                "textAlign": "justify", 
                                "marginBottom": "1rem", 
                                "fontSize": "1.2rem", 
                                "lineHeight": "1.8"
                            }
                        )
                    ], style={
                        "padding": "1rem", 
                        "backgroundColor": "var(--color-surface)", 
                        "borderRadius": "8px", 
                        "boxShadow": "var(--shadow-medium)"
                    })
                ], style={"padding": "2rem"})
            ], style={"padding": "2rem", "backgroundColor": "var(--color-background)"})
        ]),

        # ---------------- All-Inclusive Guide Tab ----------------
        dcc.Tab(label="All-Inclusive Guide", children=[
            html.Div([
                html.H2("Our Guide to Equities", className="fade-in-up", 
                        style={"textAlign": "center", "color": "var(--color-primary)"}),
                html.P(
                    [
                        "Discover how EQuity uses advanced analysis platforms to provide stock picks...",
                        html.Span("only $10", style={"fontWeight": "bold", "textDecoration": "underline", "color": "var(--color-secondary)"}),
                        "... gain access to our all-inclusive guide..."
                    ],
                    className="fade-in-up",
                    style={"textAlign": "center", "margin": "1rem", "fontSize": "1.1rem", "lineHeight": "1.8"}
                ),
                html.Div([
                    html.H3("Features Include:", style={"color": "var(--color-secondary)"}),
                    html.Ul([
                        html.Li("Data-driven stock selections leveraging AI.", className="hoverable"),
                        html.Li("Comprehensive financial and technical analyses.", className="hoverable"),
                        html.Li("Insights into the crypto market with actionable strategies.", className="hoverable"),
                    ], style={
                        "lineHeight": "1.8", 
                        "padding": "1rem", 
                        "border": "1px solid var(--color-border)", 
                        "borderRadius": "8px"
                    })
                ], style={
                    "backgroundColor": "var(--color-surface)", 
                    "padding": "1.5rem", 
                    "boxShadow": "var(--shadow-light)"
                }),
                html.Div([
                    html.Button(
                        "Buy Now", 
                        id="buy-now-button",
                        className="button-ripple",
                        style={
                            "padding": "0.75rem 1.5rem",
                            "backgroundColor": "var(--color-primary)",
                            "color": "#fff",
                            "border": "none",
                            "borderRadius": "var(--radius-base)",
                            "boxShadow": "var(--shadow-hover)"
                        }
                    )
                ], style={"textAlign": "center", "marginTop": "2rem"}),
                html.Div(
                    id="payment-confirmation", 
                    style={"marginTop": "1rem", "color": "green", "textAlign": "center"}
                )
            ], style={"padding": "2rem", "backgroundColor": "var(--color-background)"})
        ]),

        # ---------------- Fundamentals Tab ----------------
        dcc.Tab(label="Fundamentals", children=[
            html.Div([
                html.H2("Fundamental Analysis", 
                        style={"textAlign": "center", "color": "var(--color-primary)", "marginTop": "1rem"}),
                html.Div([
                    dcc.Input(
                        id='ticker-input',
                        type='text',
                        placeholder='Enter ticker symbol',
                        style={"marginRight": "1rem", "marginBottom": "1rem"}
                    ),
                    html.Button(
                        "Fetch", 
                        id='fetch-button',
                        style={"marginRight": "1rem"}
                    ),
                    html.Button(
                        "Reset", 
                        id='reset-button',
                        style={"marginRight": "1rem"}
                    ),
                    html.Button(
                        "Download CSV", 
                        id='download-button',
                        style={"marginRight": "1rem"}
                    ),
                ], style={"textAlign": "center", "marginBottom": "1rem"}),
        
                dash_table.DataTable(
                    id='metrics-table',
                    columns=FUND_COLUMNS_DEFS,
                    data=[],
                    style_table={'overflowX': 'auto', 'marginBottom': '1rem'},
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                ),
            ], style={"padding": "2rem", "backgroundColor": "var(--color-background)"})
        ]),

        # ---------------- Sector Analysis Tab ----------------
        dcc.Tab(label="Sector Analysis", children=[
            html.Div([
                html.H2("Top 5 Symbols by Sector", 
                        style={"textAlign": "center", 'marginTop': '1rem', 
                               "color": "var(--color-primary)", "animation": "fadeInUp 0.5s ease"}),
                html.Div(
                    children=[
                        html.P(
                            [
                                html.Span("How We Select Top Stocks: ", style={"fontWeight": "bold", "color": "var(--color-primary)"}),
                                "Our selection process begins with SEC filings, evaluating all stock symbols, and then we select the top 5 per sector, based on a blend of different ratios and data from earnings statements."
                            ],
                            className="fade-in-up",
                            style={
                                "textAlign": "justify", 
                                "marginBottom": "2rem", 
                                "backgroundColor": "var(--color-surface)",
                                "padding": "1rem", 
                                "borderRadius": "8px", 
                                "border": "1px solid var(--color-border)"
                            }
                        )
                    ]
                ),
                html.Div([
                    html.Div([
                        html.H3(f"{sector}", style={"color": "var(--color-primary)"}),
                        dash_table.DataTable(
                            columns=[{"name": col, "id": col} for col in df.columns],
                            data=(df.sort_values(by='VALUE RATIO', ascending=(sector == "Basic Materials"))
                                  .head(5)
                                  .to_dict('records')),
                            style_table={'marginBottom': '1rem', 'overflowX': 'auto'},
                            style_cell={'textAlign': 'center', 'padding': '5px'},
                            style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'}
                        )
                    ], style={'marginBottom': '2rem'}) for sector, df in sector_dataframes.items()
                ])
            ], style={'padding': '1rem'})
        ]),

        # ============= NEW TAB FOR SIMILAR SYMBOLS =============
        dcc.Tab(label="Similar Stocks", children=[
            html.Div([
                html.H2("Discover Similar Companies",
                    style={"textAlign": "center", "color": "var(--color-primary)", "marginTop": "1rem"}),

                html.Div([
                    dcc.Input(
                        id='similar-ticker-input',
                        type='text',
                        placeholder='Enter ticker symbol',
                        style={"marginRight": "1rem", "marginBottom": "1rem"}
                    ),
                    html.Button(
                        "Get Similar Stocks", 
                        id='similar-button',
                        style={"marginRight": "1rem"}
                    ),
                ], style={"textAlign": "center"}),

                dash_table.DataTable(
                    id='similar-table',
                    columns=[{"name": "Symbol", "id": "Symbol"}],
                    data=[],
                    style_table={'overflowX': 'auto', 'marginTop': '1rem'},
                    style_cell={'textAlign': 'center', 'padding': '5px'},
                    style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                )
            ], style={"padding": "2rem", "backgroundColor": "var(--color-background)"})
        ]),
    ])
])

# ------------------- CALLBACKS -------------------

# Callback for the new Similar Stocks tab
@app.callback(
    Output('similar-table', 'data'),
    Input('similar-button', 'n_clicks'),
    State('similar-ticker-input', 'value'),
    prevent_initial_call=True
)
def fetch_similar_stocks(n_clicks, ticker):
    """
    Uses WS1 from assets/func/similar.py to scrape a list of related symbols
    on FinViz for the given ticker.
    """
    if not ticker:
        return []

    # Instantiate WS1 and scrape
    ws1 = WS1(ticker)
    related_symbols = ws1.scrape()

    # Convert the list into the format for Dash DataTable
    data_rows = [{"Symbol": sym} for sym in related_symbols]
    return data_rows

@app.callback(
    Output("payment-confirmation", "children"),
    Input("buy-now-button", "n_clicks"),
    prevent_initial_call=True
)
def handle_buy_now(n_clicks):
    """
    Creates a Stripe checkout session and returns a link to redirect.
    """
    if dash.callback_context.triggered_id == "buy-now-button":
        try:
            session = stripe.checkout.Session.create(
                payment_method_types=["card"],
                line_items=[
                    {
                        "price": "price_1QbknF4SsxaWFVRwwkdeyLWs",  # Correct price_id
                        "quantity": 1,
                    }
                ],
                mode="payment",
                success_url="https://your-success-url.com",  # Replace with your success URL
                cancel_url="https://your-cancel-url.com",    # Replace with your cancel URL
            )
            return html.A(
                "Redirecting to Stripe Checkout...",
                href=session.url,
                target="_blank",
                style={"color": "var(--color-primary)", "fontWeight": "bold"}
            )
        except Exception as e:
            return f"Error creating checkout session: {e}"

# Callback for Fundamental Tab
@app.callback(
    Output('metrics-store', 'data'),
    Output('metrics-table', 'data'),
    [Input('fetch-button', 'n_clicks'), Input('reset-button', 'n_clicks')],
    [State('ticker-input', 'value'), State('metrics-store', 'data')],
    prevent_initial_call=True
)
def update_table(n_fetch, n_reset, ticker, current_data):
    if dash.callback_context.triggered_id == 'reset-button':
        return [], []
    elif dash.callback_context.triggered_id == 'fetch-button' and ticker:
        df = get_metrics(ticker)
        df = df[df['Metric'].isin([
            'Market Cap', 'Forward P/E', 'P/E', 'Insider Own', 'Short Interest',
            'Income', 'Sales', 'ROE', 'ROA', 'Beta', 'Employees', 'Sales Y/Y TTM'
        ])]

        if df.empty:
            metrics_list = [None] * len(FUND_COLUMNS[1:])
        else:
            metrics_list = df['Value'].apply(convert_to_number).tolist()
        
        row_dict = {col: metrics_list[i] if i < len(metrics_list) else None
                    for i, col in enumerate(FUND_COLUMNS[1:])}
        row_dict['Ticker'] = ticker.upper()

        updated_data = current_data + [row_dict]
        return updated_data, updated_data
    return dash.no_update, dash.no_update

@app.callback(
    Output('download-csv', 'data'),
    Input('download-button', 'n_clicks'),
    State('metrics-store', 'data'),
    prevent_initial_call=True
)
def download_csv(n_clicks, data):
    if data:
        df = pd.DataFrame(data)
        return dcc.send_data_frame(df.to_csv, "fundamental_analysis.csv")
    return None

if __name__ == '__main__':
    app.run_server(debug=True, port=8150)

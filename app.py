import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('spanish'))

def clean_text(text):
    words = [word.lower() for word in str(text).split() if word.isalpha()]
    words = [word for word in words if word not in STOPWORDS]
    return ' '.join(words)

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positivo'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negativo'
    else:
        return 'Neutro'

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    buf = BytesIO()
    wc.to_image().save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{data}"

app = dash.Dash(__name__)
server = app.server  # Para gunicorn

app.layout = html.Div([
    html.H1("Análisis de Opiniones de Clientes"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Sube tu archivo CSV'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Hr(),
    html.Div(id='extra-analysis')
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(BytesIO(decoded))
    if 'opinion' not in df.columns or len(df) < 20:
        return html.Div([
            html.H5('El archivo debe tener una columna llamada "opinion" y al menos 20 filas.')
        ])
    df['cleaned'] = df['opinion'].apply(clean_text)

    # Nube de palabras
    all_text = ' '.join(df['cleaned'])
    wc_img = generate_wordcloud(all_text)

    # Top 10 palabras
    words = all_text.split()
    common_words = Counter(words).most_common(10)
    words_plot, counts_plot = zip(*common_words) if common_words else ([], [])
    fig_bar = px.bar(x=words_plot, y=counts_plot, labels={'x':'Palabra','y':'Frecuencia'}, title='Top 10 palabras más frecuentes')

    # Sentimiento
    df['sentimiento'] = df['opinion'].apply(get_sentiment)
    fig_pie = px.pie(df, names='sentimiento', title='Porcentaje de opiniones por sentimiento')

    # Tabla de opiniones y sentimiento
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in ['opinion', 'sentimiento']],
        data=df[['opinion', 'sentimiento']].to_dict('records'),
        style_table={'overflowX': 'auto'},
        page_size=10
    )

    return html.Div([
        html.H3('Nube de palabras'),
        html.Img(src=wc_img, style={'width':'80%'}),
        html.H3('Top 10 palabras'),
        dcc.Graph(figure=fig_bar),
        html.H3('Opiniones y sentimiento'),
        table,
        html.H3('Distribución de sentimientos'),
        dcc.Graph(figure=fig_pie),
        html.Hr(),
        html.H4('Análisis extra'),
        html.Div([
            dcc.Textarea(id='new-comment', placeholder='Escribe un comentario nuevo...', style={'width':'100%'}),
            html.Button('Analizar comentario', id='analyze-btn', n_clicks=0),
            html.Div(id='analysis-result'),
        ]),
        html.Br(),
        html.Button('Resumen de todas las opiniones', id='summary-btn', n_clicks=0),
        html.Div(id='summary-result'),
        dcc.Store(id='stored-data', data=df.to_dict('records'))
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        return parse_contents(contents, filename)
    return html.Div()

@app.callback(
    Output('analysis-result', 'children'),
    Input('analyze-btn', 'n_clicks'),
    State('new-comment', 'value')
)
def analyze_new_comment(n_clicks, comment):
    if n_clicks > 0 and comment:
        sentiment = get_sentiment(comment)
        resumen = comment if len(comment.split()) <= 20 else " ".join(comment.split()[:20]) + "..."
        return html.Div([
            html.P(f"Sentimiento: {sentiment}"),
            html.P(f"Resumen: {resumen}")
        ])
    return ""

@app.callback(
    Output('summary-result', 'children'),
    Input('summary-btn', 'n_clicks'),
    State('stored-data', 'data')
)
def summarize_all(n_clicks, data):
    if n_clicks > 0 and data:
        df = pd.DataFrame(data)
        all_text = " ".join(df['opinion'])
        resumen = all_text if len(all_text.split()) <= 60 else " ".join(all_text.split()[:60]) + "..."
        return html.Div([
            html.P("Resumen de opiniones:"),
            html.P(resumen)
        ])
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)

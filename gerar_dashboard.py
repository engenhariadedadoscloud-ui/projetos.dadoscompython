# -*- coding: utf-8 -*-
"""
Dashboard Executivo — MercadoMais S.A.
Gera um arquivo HTML auto-contido para apresentacao a diretoria.
Execute: py -3 gerar_dashboard.py
"""
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT   = Path(__file__).parent.parent.parent
DATA   = ROOT / "data"
OUTPUT = Path(__file__).parent / "dashboard.html"

# ── Carregamento ──────────────────────────────────────────────────────────────
cliente     = pd.read_csv(DATA / "cliente.csv")
pedido      = pd.read_csv(DATA / "pedido.csv")
item_pedido = pd.read_csv(DATA / "item_pedido.csv")
entrega     = pd.read_csv(DATA / "entrega.csv")
reclamacao  = pd.read_csv(DATA / "reclamacao.csv")
campanha    = pd.read_csv(DATA / "campanha.csv")
interacao   = pd.read_csv(DATA / "interacao_campanha.csv")

pedido["data_pedido"]    = pd.to_datetime(pedido["data_pedido"])
entrega["data_entregue"] = pd.to_datetime(entrega["data_entregue"], errors="coerce")

# ── KPIs ──────────────────────────────────────────────────────────────────────
ped_ok  = pedido[pedido["status"] == "Entregue"]
ped_can = pedido[pedido["status"] == "Cancelado"]
ped_dev = pedido[pedido["status"] == "Devolvido"]

receita_total = ped_ok["valor_total"].sum()
ticket_medio  = ped_ok["valor_total"].mean()
taxa_cancel   = len(ped_can) / len(pedido) * 100
taxa_devol    = len(ped_dev) / len(pedido) * 100
satisf_media  = cliente["score_satisfacao"].mean()
taxa_resolucao= reclamacao["resolvido"].mean() * 100

ent_ped = entrega.merge(pedido[["id_pedido","frete"]], on="id_pedido")
ent_ped["deficit"] = ent_ped["custo_logistico"] - ent_ped["frete"]
deficit_total = ent_ped["deficit"].sum()

roi_camp = (
    interacao.merge(campanha, on="id_campanha")
    .groupby(["id_campanha","investimento"])
    .agg(receita_gerada=("valor_compra","sum")).reset_index()
)
roi_camp["roi"] = (roi_camp["receita_gerada"] - roi_camp["investimento"]) / roi_camp["investimento"] * 100
roi_medio = roi_camp["roi"].mean()

# ── Cores e layout padrao ─────────────────────────────────────────────────────
COR_OK    = "#27ae60"
COR_ALERT = "#e74c3c"
COR_WARN  = "#f39c12"
COR_BLUE  = "#2980b9"

BASE_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="#f8f9fa",
    font=dict(family="Segoe UI, Arial", size=12, color="#2c3e50"),
    margin=dict(t=50, l=50, r=20, b=50),
    title_font=dict(size=14, color="#1a1a2e"),
)

def to_div(fig, div_id):
    fig.update_layout(**BASE_LAYOUT)
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

# ── Grafico 1: Evolucao temporal ──────────────────────────────────────────────
vendas_mes = (
    ped_ok.assign(ano_mes=lambda d: d["data_pedido"].dt.to_period("M"))
    .groupby("ano_mes")
    .agg(receita=("valor_total","sum"), pedidos=("id_pedido","count"))
    .reset_index()
)
vendas_mes["ano_mes"] = vendas_mes["ano_mes"].astype(str)

fig_temporal = make_subplots(specs=[[{"secondary_y": True}]])
fig_temporal.add_trace(
    go.Scatter(x=vendas_mes["ano_mes"], y=vendas_mes["receita"], name="Receita (R$)",
               fill="tozeroy", line=dict(color=COR_BLUE, width=2.5)),
    secondary_y=False)
fig_temporal.add_trace(
    go.Bar(x=vendas_mes["ano_mes"], y=vendas_mes["pedidos"], name="Pedidos",
           marker_color=COR_OK, opacity=0.45),
    secondary_y=True)
fig_temporal.update_layout(title="Evolucao da Receita Mensal", hovermode="x unified",
                            legend=dict(orientation="h", y=1.12))
fig_temporal.update_xaxes(tickangle=45)
fig_temporal.update_yaxes(title_text="Receita (R$)", secondary_y=False)
fig_temporal.update_yaxes(title_text="Pedidos",      secondary_y=True, showgrid=False)

# ── Grafico 2: Receita por estado ─────────────────────────────────────────────
por_estado = (
    ped_ok.merge(cliente[["id_cliente","estado"]], on="id_cliente")
    .groupby("estado")
    .agg(receita=("valor_total","sum"), ticket=("valor_total","mean"))
    .reset_index().sort_values("receita", ascending=True)
)

fig_regional = go.Figure(go.Bar(
    x=por_estado["receita"], y=por_estado["estado"], orientation="h",
    marker_color=COR_BLUE,
    text=[f"R$ {v:,.0f}" for v in por_estado["receita"]],
    textposition="outside",
))
fig_regional.update_layout(title="Receita Total por Estado",
                            xaxis_title="Receita (R$)", yaxis_title="")

# ── Grafico 3: Status dos pedidos (donut) ─────────────────────────────────────
status_cnt  = pedido["status"].value_counts().reset_index()
status_cnt.columns = ["status","qtd"]
cores_status = [{"Entregue": COR_OK, "Cancelado": COR_ALERT,
                  "Devolvido": COR_WARN, "Em transporte": COR_BLUE}.get(s,"#95a5a6")
                 for s in status_cnt["status"]]

fig_status = go.Figure(go.Pie(
    labels=status_cnt["status"], values=status_cnt["qtd"],
    hole=0.55, marker_colors=cores_status,
    textinfo="percent+label",
))
fig_status.update_layout(title="Status dos Pedidos", showlegend=False)

# ── Grafico 4: Logistica ──────────────────────────────────────────────────────
transp = (
    ent_ped.groupby("transportadora")
    .agg(custo=("custo_logistico","mean"), frete=("frete","mean"))
    .reset_index()
)
transp["deficit"] = transp["custo"] - transp["frete"]
transp = transp.sort_values("deficit", ascending=False)

fig_logistica = go.Figure()
fig_logistica.add_trace(go.Bar(name="Custo Logistico", x=transp["transportadora"],
                                y=transp["custo"], marker_color=COR_ALERT))
fig_logistica.add_trace(go.Bar(name="Frete Cobrado", x=transp["transportadora"],
                                y=transp["frete"], marker_color=COR_OK))
fig_logistica.update_layout(title="Custo Logistico vs Frete Cobrado por Transportadora",
                              barmode="group", xaxis_title="", yaxis_title="Valor Medio (R$)",
                              legend=dict(orientation="h", y=1.12))

# ── Grafico 5: Reclamacoes ────────────────────────────────────────────────────
cat_rec = reclamacao["categoria"].value_counts().reset_index()
cat_rec.columns = ["categoria","qtd"]

fig_reclamacoes = go.Figure(go.Bar(
    x=cat_rec["qtd"], y=cat_rec["categoria"], orientation="h",
    marker_color=COR_ALERT,
    text=cat_rec["qtd"], textposition="outside",
))
fig_reclamacoes.update_layout(title="Reclamacoes por Categoria",
                               xaxis_title="Quantidade", yaxis_title="")

# ── Grafico 6: ROI das campanhas ──────────────────────────────────────────────
roi_det = (
    interacao.merge(campanha, on="id_campanha")
    .groupby(["id_campanha","nome","canal","investimento"])
    .agg(receita_gerada=("valor_compra","sum"),
         total_views=("visualizou","sum"),
         total_compras=("comprou","sum"))
    .reset_index()
)
roi_det["roi"] = ((roi_det["receita_gerada"] - roi_det["investimento"]) / roi_det["investimento"] * 100).round(1)
roi_det["nome_curto"] = roi_det["nome"].str.replace(r"Campanha \d+ - ", "", regex=True)
roi_det = roi_det.sort_values("roi")

fig_roi = go.Figure(go.Bar(
    x=roi_det["roi"], y=roi_det["nome_curto"], orientation="h",
    marker_color=[COR_ALERT if r < 0 else COR_OK for r in roi_det["roi"]],
    text=[f"{r:.1f}%" for r in roi_det["roi"]], textposition="outside",
))
fig_roi.add_vline(x=0, line_dash="dash", line_color="gray")
fig_roi.update_layout(title="ROI por Campanha de Marketing",
                       xaxis_title="ROI (%)", yaxis_title="")

# ── Grafico 7: Segmentacao K-Means ───────────────────────────────────────────
data_ref = pedido["data_pedido"].max()
rfm = (
    ped_ok.groupby("id_cliente")
    .agg(
        recencia   = ("data_pedido", lambda x: (data_ref - x.max()).days),
        frequencia = ("id_pedido",   "count"),
        monetario  = ("valor_total", "sum"),
    ).reset_index()
    .merge(cliente[["id_cliente","renda_mensal"]], on="id_cliente")
)

X_scaled = StandardScaler().fit_transform(rfm[["recencia","frequencia","monetario","renda_mensal"]])
rfm["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)

perfil_mon = rfm.groupby("cluster")["monetario"].mean()
lbl = {
    perfil_mon.idxmax(): "Alto Valor",
    perfil_mon.idxmin(): "Baixo Engajamento",
    [c for c in perfil_mon.index if c not in (perfil_mon.idxmax(), perfil_mon.idxmin())][0]: "Potencial",
}
rfm["segmento"] = rfm["cluster"].map(lbl)
cores_seg = {"Alto Valor": COR_OK, "Potencial": COR_BLUE, "Baixo Engajamento": COR_ALERT}

fig_seg = go.Figure()
for seg, cor in cores_seg.items():
    sub = rfm[rfm["segmento"] == seg]
    fig_seg.add_trace(go.Scatter(
        x=sub["frequencia"], y=sub["monetario"],
        mode="markers", name=f"{seg} ({len(sub)})",
        marker=dict(color=cor, size=11, opacity=0.85, line=dict(color="white", width=1))
    ))
fig_seg.update_layout(
    title="Segmentacao de Clientes (K-Means, K=3)",
    xaxis_title="Frequencia (pedidos)", yaxis_title="Valor Monetario (R$)",
    legend=dict(orientation="h", y=1.12),
)

# ── Monta HTML ────────────────────────────────────────────────────────────────
def kpi_card(titulo, valor, sub, cor, alerta=False):
    borda = COR_ALERT if alerta else cor
    return f"""
    <div class="kpi-card" style="border-left:5px solid {borda}">
      <div class="kpi-lbl">{titulo}</div>
      <div class="kpi-val" style="color:{borda}">{valor}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

kpis = "".join([
    kpi_card("Receita Efetiva",      f"R$ {receita_total:,.2f}", "Pedidos entregues",       COR_BLUE),
    kpi_card("Ticket Medio",         f"R$ {ticket_medio:,.2f}",  "Por pedido entregue",      COR_BLUE),
    kpi_card("Taxa Cancelamento",    f"{taxa_cancel:.1f}%",       "Do total de pedidos",      COR_WARN, taxa_cancel > 10),
    kpi_card("Devolucoes",           f"{taxa_devol:.1f}%",        "Do total de pedidos",      COR_WARN, taxa_devol > 5),
    kpi_card("Deficit Logistico",    f"R$ {deficit_total:,.2f}",  "Custo menos frete cobrado",COR_ALERT, True),
    kpi_card("ROI Medio Campanhas",  f"{roi_medio:.1f}%",         "Media das campanhas",      COR_OK if roi_medio >= 0 else COR_ALERT, roi_medio < 0),
    kpi_card("Satisfacao Media",     f"{satisf_media:.1f} / 10",  "Score dos clientes",       COR_OK if satisf_media >= 6 else COR_ALERT),
    kpi_card("Resolucao Reclamacoes",f"{taxa_resolucao:.1f}%",    "Reclamacoes resolvidas",   COR_OK if taxa_resolucao >= 70 else COR_WARN),
])

today = date.today().strftime("%d/%m/%Y")

html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Dashboard Executivo — MercadoMais S.A.</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2c3e50}}

    header{{
      background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
      color:#fff;padding:22px 40px;
      display:flex;justify-content:space-between;align-items:center;
      box-shadow:0 3px 10px rgba(0,0,0,.35)
    }}
    header h1{{font-size:1.5rem;font-weight:700;letter-spacing:.5px}}
    header .meta{{font-size:.78rem;opacity:.65;margin-top:5px}}
    .badge{{
      background:#e94560;color:#fff;
      padding:7px 16px;border-radius:20px;
      font-size:.78rem;font-weight:700;letter-spacing:1px
    }}

    .wrap{{max-width:1400px;margin:0 auto;padding:20px 24px}}

    .sec-title{{
      font-size:.82rem;font-weight:700;color:#1a1a2e;
      text-transform:uppercase;letter-spacing:2px;
      margin:28px 0 12px;padding-bottom:8px;
      border-bottom:2px solid #1a1a2e
    }}

    .kpi-grid{{
      display:grid;
      grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
      gap:12px;margin-bottom:4px
    }}
    .kpi-card{{
      background:#fff;border-radius:8px;padding:14px 16px;
      box-shadow:0 2px 8px rgba(0,0,0,.07)
    }}
    .kpi-lbl{{font-size:.67rem;font-weight:700;color:#7f8c8d;text-transform:uppercase;letter-spacing:.8px}}
    .kpi-val{{font-size:1.45rem;font-weight:800;margin:6px 0 3px}}
    .kpi-sub{{font-size:.68rem;color:#aab}}

    .g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}}
    .g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}}
    .card{{background:#fff;border-radius:8px;padding:6px;box-shadow:0 2px 8px rgba(0,0,0,.07)}}
    .full{{grid-column:1/-1}}

    footer{{
      text-align:center;padding:18px;
      font-size:.72rem;color:#aab;margin-top:10px
    }}
  </style>
</head>
<body>
<header>
  <div>
    <h1>MercadoMais S.A. &mdash; Dashboard Executivo</h1>
    <div class="meta">Operacao Retomada &nbsp;|&nbsp; Gerado em {today}</div>
  </div>
  <div class="badge">&#9888; CRISE ATIVA</div>
</header>

<div class="wrap">

  <div class="sec-title">Indicadores Estrategicos</div>
  <div class="kpi-grid">{kpis}</div>

  <div class="sec-title">Evolucao Temporal</div>
  <div class="card full">{to_div(fig_temporal, "temporal")}</div>

  <div class="sec-title">Comparativos Regionais</div>
  <div class="g2">
    <div class="card">{to_div(fig_regional, "regional")}</div>
    <div class="card">{to_div(fig_status, "status")}</div>
  </div>

  <div class="sec-title">Perdas e Riscos</div>
  <div class="g2">
    <div class="card">{to_div(fig_logistica, "logistica")}</div>
    <div class="card">{to_div(fig_reclamacoes, "reclamacoes")}</div>
  </div>

  <div class="sec-title">Resultados dos Modelos Analiticos</div>
  <div class="g2">
    <div class="card">{to_div(fig_seg, "seg")}</div>
    <div class="card">{to_div(fig_roi, "roi")}</div>
  </div>

</div>

<footer>MercadoMais S.A. &nbsp;&bull;&nbsp; Hackathon Operacao Retomada &nbsp;&bull;&nbsp; Confidencial</footer>
</body>
</html>"""

OUTPUT.write_text(html, encoding="utf-8")
print(f"Dashboard gerado em: {OUTPUT}")

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Bitcoin", 
    page_icon="₿", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #f7931e, #ffb84d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares
@st.cache_data
def cargar_datos_ejemplo():
    """Genera datos de ejemplo para demostración"""
    np.random.seed(42)
    fechas = pd.date_range(start='2023-01-01', end='2024-11-01', freq='D')
    days = np.arange(len(fechas))
    
    # Simular precio de Bitcoin
    trend = 35000 + 20000 * np.sin(days / 365.25 * 2 * np.pi) + days * 25
    seasonal_volatility = 1 + 0.3 * np.sin(days / 30 * 2 * np.pi)
    weekly_pattern = 1000 * np.sin(days / 7 * 2 * np.pi)
    noise = np.random.normal(0, 2000 * seasonal_volatility, len(fechas))
    
    precios = trend + weekly_pattern + noise
    precios = np.maximum(precios, 20000)
    
    df = pd.DataFrame({
        'ds': fechas,
        'y': precios
    })
    
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df

@st.cache_data(ttl=1800)
def obtener_datos_coingecko(dias_historicos=365):
    """Obtiene datos históricos de Bitcoin de CoinGecko"""
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd', 
            'days': str(min(dias_historicos, 365)),
            'interval': 'daily'
        }
        
        with st.spinner(f"🔄 Obteniendo datos de CoinGecko..."):
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
        
        data = response.json()
        precios = data['prices']
        df = pd.DataFrame(precios, columns=['timestamp', 'price'])
        
        df['ds'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
        df['y'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[['ds', 'y']].dropna()
        
        st.success(f"✅ {len(df)} registros obtenidos de CoinGecko")
        return df
        
    except Exception as e:
        st.error(f"❌ Error con CoinGecko: {str(e)}")
        return None

def limpiar_datos(df):
    """Limpia y valida los datos para Prophet"""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_localize(None)
    
    df = df.dropna()
    df = df[df['y'] > 0]
    df = df.sort_values('ds').reset_index(drop=True)
    df = df.drop_duplicates(subset=['ds'], keep='last')
    
    return df

def crear_modelo_prophet(df, params):
    """Crea y entrena el modelo Prophet"""
    try:
        modelo = Prophet(
            daily_seasonality=params['estacionalidad_diaria'] and len(df) > 60,
            weekly_seasonality=params['estacionalidad_semanal'],
            yearly_seasonality=params['estacionalidad_anual'] and len(df) > 180,
            interval_width=params['intervalo_confianza']
        )
        
        with st.spinner("🧠 Entrenando modelo Prophet..."):
            modelo.fit(df)
        
        return modelo
        
    except Exception as e:
        st.error(f"❌ Error al entrenar modelo: {str(e)}")
        return None

# Título principal
st.markdown('<h1 class="main-header">₿ Predicción del Precio de Bitcoin</h1>', unsafe_allow_html=True)

# Sidebar para controles
with st.sidebar:
    st.header("⚙️ Configuración")
    
    opcion = st.radio(
        "Origen de los datos:",
        ["🌐 API CoinGecko", "📂 Subir CSV", "🎲 Datos de ejemplo"]
    )
    
    st.subheader("🔧 Parámetros del Modelo")
    dias = st.slider("Días a predecir:", min_value=7, max_value=365, value=30, step=1)
    
    with st.expander("⚙️ Configuración Avanzada"):
        intervalo_confianza = st.slider("Intervalo de confianza:", 0.80, 0.99, 0.95, 0.01)
        estacionalidad_anual = st.checkbox("Estacionalidad anual", value=True)
        estacionalidad_semanal = st.checkbox("Estacionalidad semanal", value=True)
        estacionalidad_diaria = st.checkbox("Estacionalidad diaria", value=False)
        
        if opcion == "🌐 API CoinGecko":
            dias_historicos = st.slider("Días históricos:", 30, 365, 365, 1)

# Cargar datos según la opción seleccionada
df = None
fuente_datos = ""

if opcion == "📂 Subir CSV":
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Detectar columnas automáticamente
            columnas_map = {}
            for col in df.columns:
                if any(x in col.lower() for x in ['date', 'time', 'fecha', 'ds']):
                    columnas_map['ds'] = col
                    break
            
            for col in df.columns:
                if any(x in col.lower() for x in ['close', 'price', 'precio', 'y', 'value']):
                    columnas_map['y'] = col
                    break
            
            if 'ds' not in columnas_map or 'y' not in columnas_map:
                st.error("❌ No se encontraron columnas válidas de fecha y precio")
                st.stop()
            
            df = df.rename(columns={columnas_map['ds']: 'ds', columnas_map['y']: 'y'})
            df = limpiar_datos(df)
            fuente_datos = "CSV subido por usuario"
            
            if df is not None and len(df) >= 10:
                st.success(f"✅ CSV cargado: {len(df)} registros válidos")
            else:
                st.error("❌ Datos insuficientes")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")
            st.stop()
    else:
        st.info("⬆️ Por favor, sube un archivo CSV para continuar")
        st.stop()

elif opcion == "🌐 API CoinGecko":
    df = obtener_datos_coingecko(dias_historicos)
    
    if df is not None and len(df) >= 10:
        fuente_datos = "CoinGecko API"
    else:
        st.warning("⚠️ CoinGecko no disponible. Usando datos de ejemplo...")
        df = cargar_datos_ejemplo()
        fuente_datos = "Datos de ejemplo"

else:  # Datos de ejemplo
    df = cargar_datos_ejemplo()
    fuente_datos = "Datos de ejemplo generados"
    st.info(f"🎲 {fuente_datos}: {len(df)} registros")

# Verificar datos válidos
if df is None or len(df) < 10:
    st.error("❌ No hay suficientes datos para entrenar el modelo")
    st.stop()

# Parámetros del modelo
params = {
    'intervalo_confianza': intervalo_confianza,
    'estacionalidad_anual': estacionalidad_anual,
    'estacionalidad_semanal': estacionalidad_semanal,
    'estacionalidad_diaria': estacionalidad_diaria
}

# Estadísticas básicas
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Registros", f"{len(df):,}")

with col2:
    precio_actual = df['y'].iloc[-1]
    if len(df) > 1:
        precio_anterior = df['y'].iloc[-2]
        delta = precio_actual - precio_anterior
        delta_pct = (delta / precio_anterior) * 100
        st.metric("💰 Precio Actual", f"${precio_actual:,.2f}", f"{delta:+,.2f} ({delta_pct:+.1f}%)")
    else:
        st.metric("💰 Precio Actual", f"${precio_actual:,.2f}")

with col3:
    precio_max = df['y'].max()
    st.metric("📈 Máximo", f"${precio_max:,.2f}")

with col4:
    precio_min = df['y'].min()
    st.metric("📉 Mínimo", f"${precio_min:,.2f}")

# Información de la fuente
st.markdown(f"""
<div class="warning-box">
    <strong>📊 Fuente:</strong> {fuente_datos} | 
    <strong>📅 Período:</strong> {df['ds'].min().strftime('%Y-%m-%d')} a {df['ds'].max().strftime('%Y-%m-%d')} 
    ({(df['ds'].max() - df['ds'].min()).days} días)
</div>
""", unsafe_allow_html=True)

# Pestañas principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Datos Históricos", 
    "🔮 Predicción", 
    "📈 Componentes", 
    "🎯 Consulta Específica"
])

with tab1:
    st.subheader("📊 Análisis de Datos Históricos")
    
    # Gráfico histórico
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Scatter(
        x=df['ds'], 
        y=df['y'],
        mode='lines',
        name='Precio Histórico',
        line=dict(color='#f7931e', width=2),
        hovertemplate='<b>Fecha:</b> %{x}<br><b>Precio:</b> $%{y:,.2f}<extra></extra>'
    ))
    
    # Promedio móvil si hay suficientes datos
    if len(df) > 30:
        df_ma = df.copy()
        df_ma['ma_30'] = df_ma['y'].rolling(window=30).mean()
        
        fig_hist.add_trace(go.Scatter(
            x=df_ma['ds'], 
            y=df_ma['ma_30'],
            mode='lines',
            name='MA 30d',
            line=dict(color='#e74c3c', width=1, dash='dash'),
            hovertemplate='<b>MA 30d:</b> $%{y:,.2f}<extra></extra>'
        ))
    
    fig_hist.update_layout(
        title="Precio Histórico de Bitcoin",
        xaxis_title="Fecha",
        yaxis_title="Precio (USD)",
        template="plotly_white",
        height=450
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Estadísticas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Estadísticas")
        stats = df['y'].describe()
        volatilidad = df['y'].pct_change().std() * np.sqrt(365) * 100
        
        stats_df = pd.DataFrame({
            'Métrica': ['Media', 'Mediana', 'Desv. Estándar', 'Volatilidad Anual'],
            'Valor': [
                f"${stats['mean']:,.2f}",
                f"${stats['50%']:,.2f}",
                f"${stats['std']:,.2f}",
                f"{volatilidad:.1f}%"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🔍 Datos Recientes")
        df_recent = df.tail(10).copy()
        df_recent['ds'] = df_recent['ds'].dt.strftime('%Y-%m-%d')
        df_recent['y'] = df_recent['y'].round(2)
        
        df_recent = df_recent.rename(columns={'ds': 'Fecha', 'y': 'Precio (USD)'})
        st.dataframe(df_recent, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🔮 Predicción con Prophet")
    
    # Entrenar modelo
    modelo = crear_modelo_prophet(df, params)
    
    if modelo is None:
        st.error("❌ No se pudo entrenar el modelo")
        st.stop()
    
    try:
        # Crear predicciones
        future = modelo.make_future_dataframe(periods=dias)
        forecast = modelo.predict(future)
        
        # Gráfica de predicción
        fig_pred = plot_plotly(modelo, forecast)
        fig_pred.update_layout(
            title=f"Predicción de Bitcoin - Próximos {dias} días",
            height=500,
            template="plotly_white"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Métricas de predicción
        prediccion_futura = forecast.tail(dias)
        precio_estimado = prediccion_futura['yhat'].iloc[-1]
        precio_actual = df['y'].iloc[-1]
        cambio_estimado = precio_estimado - precio_actual
        porcentaje_cambio = (cambio_estimado / precio_actual) * 100
        
        precio_min_est = prediccion_futura['yhat_lower'].iloc[-1]
        precio_max_est = prediccion_futura['yhat_upper'].iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Precio en {} días".format(dias), f"${precio_estimado:,.2f}")
        with col2:
            st.metric("📊 Cambio Estimado", f"${cambio_estimado:+,.2f}")
        with col3:
            st.metric("📈 Cambio %", f"{porcentaje_cambio:+.1f}%")
        with col4:
            rango = precio_max_est - precio_min_est
            st.metric("🎯 Rango Confianza", f"${rango:,.0f}")
        
        # Resumen de predicción
        st.subheader("📋 Resumen de Predicción")
        
        summary_data = {
            'Métrica': [
                'Precio Actual',
                'Precio Estimado',
                'Límite Inferior',
                'Límite Superior',
                'Cambio Absoluto',
                'Cambio Porcentual'
            ],
            'Valor': [
                f"${precio_actual:,.2f}",
                f"${precio_estimado:,.2f}",
                f"${precio_min_est:,.2f}",
                f"${precio_max_est:,.2f}",
                f"${cambio_estimado:+,.2f}",
                f"{porcentaje_cambio:+.1f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")

with tab3:
    st.subheader("📈 Componentes del Modelo")
    
    if 'modelo' in locals() and 'forecast' in locals():
        try:
            fig_comp = plot_components_plotly(modelo, forecast)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Análisis de componentes
            st.subheader("🔢 Análisis de Componentes")
            
            trend_contribution = forecast['trend'].iloc[-1] - forecast['trend'].iloc[0]
            
            if abs(trend_contribution) > 1000:
                tendencia_desc = "📈 Alcista" if trend_contribution > 0 else "📉 Bajista"
                st.markdown(f"**Tendencia General:** {tendencia_desc} (${trend_contribution:+,.0f})")
            
            if 'yearly' in forecast.columns:
                yearly_range = forecast['yearly'].max() - forecast['yearly'].min()
                st.markdown(f"**Estacionalidad Anual:** Impacto de ±${yearly_range/2:,.0f}")
            
            if 'weekly' in forecast.columns:
                weekly_range = forecast['weekly'].max() - forecast['weekly'].min()
                st.markdown(f"**Estacionalidad Semanal:** Impacto de ±${weekly_range/2:,.0f}")
            
        except Exception as e:
            st.error(f"❌ Error al mostrar componentes: {str(e)}")
    else:
        st.warning("⚠️ Primero ejecuta la predicción en la pestaña anterior")

with tab4:
    st.subheader("🎯 Consulta de Predicción Específica")
    
    if 'forecast' in locals():
        fecha_min = df['ds'].min().date()
        fecha_max = (df['ds'].max() + timedelta(days=dias)).date()
        
        fecha_consulta = st.date_input(
            "Selecciona una fecha:",
            value=min(datetime.now().date(), fecha_max),
            min_value=fecha_min,
            max_value=fecha_max
        )
        
        # Buscar predicción para la fecha
        pred_fecha = forecast[forecast['ds'].dt.date == fecha_consulta]
        
        if not pred_fecha.empty:
            valor = pred_fecha['yhat'].values[0]
            valor_min = pred_fecha['yhat_lower'].values[0]
            valor_max = pred_fecha['yhat_upper'].values[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Predicción", f"${valor:,.2f}")
            with col2:
                st.metric("📉 Límite Inferior", f"${valor_min:,.2f}")
            with col3:
                st.metric("📈 Límite Superior", f"${valor_max:,.2f}")
            
            # Análisis de la predicción
            incertidumbre = ((valor_max - valor_min) / valor) * 100
            
            st.markdown(f"""
            **📊 Análisis:**
            - **Incertidumbre:** ±{incertidumbre:.1f}% del valor predicho
            - **Rango de precios:** ${valor_min:,.0f} - ${valor_max:,.0f}
            """)
            
            # Verificar si es predicción futura
            if fecha_consulta > df['ds'].max().date():
                st.success(f"🔮 Predicción futura para {fecha_consulta}")
                dias_futuro = (fecha_consulta - df['ds'].max().date()).days
                st.info(f"📅 {dias_futuro} días en el futuro")
            else:
                precio_real = df[df['ds'].dt.date == fecha_consulta]['y'].values
                if len(precio_real) > 0:
                    diferencia = abs(valor - precio_real[0])
                    st.info(f"📊 Precio real: ${precio_real[0]:,.2f} | Diferencia: ${diferencia:,.2f}")
            
            # Gráfico de la consulta
            fecha_consulta_dt = pd.to_datetime(fecha_consulta)
            
            # Calcular rango para el gráfico
            fecha_inicio = max(fecha_consulta_dt - timedelta(days=30), forecast['ds'].min())
            fecha_fin = min(fecha_consulta_dt + timedelta(days=30), forecast['ds'].max())
            
            forecast_periodo = forecast[
                (forecast['ds'] >= fecha_inicio) & 
                (forecast['ds'] <= fecha_fin)
            ]
            
            df_periodo = df[
                (df['ds'] >= fecha_inicio) & 
                (df['ds'] <= fecha_fin)
            ]
            
            fig_consulta = go.Figure()
            
            # Datos históricos
            if not df_periodo.empty:
                fig_consulta.add_trace(go.Scatter(
                    x=df_periodo['ds'],
                    y=df_periodo['y'],
                    mode='lines+markers',
                    name='Precio Real',
                    line=dict(color='#f7931e', width=2),
                    marker=dict(size=4)
                ))
            
            # Predicción futura
            forecast_futuro = forecast_periodo[forecast_periodo['ds'] > df['ds'].max()]
            if not forecast_futuro.empty:
                fig_consulta.add_trace(go.Scatter(
                    x=forecast_futuro['ds'],
                    y=forecast_futuro['yhat'],
                    mode='lines',
                    name='Predicción',
                    line=dict(color='#ff6b35', width=2, dash='dot')
                ))
                
                # Banda de confianza
                fig_consulta.add_trace(go.Scatter(
                    x=forecast_futuro['ds'].tolist() + forecast_futuro['ds'].iloc[::-1].tolist(),
                    y=forecast_futuro['yhat_upper'].tolist() + forecast_futuro['yhat_lower'].iloc[::-1].tolist(),
                    fill='tonexty',
                    fillcolor='rgba(255, 107, 53, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalo de Confianza'
                ))
            
            # Punto consultado
            fig_consulta.add_trace(go.Scatter(
                x=[fecha_consulta_dt],
                y=[valor],
                mode='markers',
                name=f'Consulta: {fecha_consulta}',
                marker=dict(color='red', size=15, symbol='star')
            ))
            
            # Líneas de referencia
            fig_consulta.add_vline(x=fecha_consulta_dt, line_dash="dash", line_color="red", opacity=0.7)
            fig_consulta.add_vline(x=df['ds'].max(), line_dash="solid", line_color="gray", opacity=0.5)
            
            fig_consulta.update_layout(
                title=f"Análisis detallado para {fecha_consulta}",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig_consulta, use_container_width=True)
        else:
            st.warning("⚠️ Fecha fuera del rango del modelo")
    else:
        st.warning("⚠️ Primero ejecuta la predicción en la pestaña 'Predicción'")

# Footer simplificado
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>⚠️ Aviso:</strong> Esta aplicación es solo para fines educativos. 
    Las predicciones NO constituyen asesoramiento financiero.</p>
    <p><small>Desarrollado con Streamlit y Facebook Prophet</small></p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Arbolado de Rivera - Dashboard",
    page_icon="🌳",
    layout="wide",
)

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495e;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stProgress .st-bp {
        background-color: #3498db;
    }
    .filter-section {
        background-color: #edf2f7;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Application header
st.markdown('<div class="main-header">🌳 Arbolado de Rivera - Dashboard</div>', unsafe_allow_html=True)

# Load and preprocess the data
@st.cache_data
def load_data():
    try:
        # Try to find the file in different possible locations
        file_paths = [
            'Arbolado de Rivera - Prueba - ALVARO (3).xlsx',  # Current directory
            './Arbolado de Rivera - Prueba - ALVARO (3).xlsx',  # Explicit current directory
            '../Arbolado de Rivera - Prueba - ALVARO (3).xlsx',  # Parent directory
            '/workspace/uploads/Arbolado de Rivera - Prueba - ALVARO (3).xlsx',  # Original path
        ]
        
        # Try each path until one works
        data = None
        for path in file_paths:
            try:
                st.info(f"Trying to load from: {path}")
                data = pd.read_excel(path, skiprows=1)
                st.success(f"Successfully loaded data from: {path}")
                break
            except Exception as e:
                st.warning(f"Could not load from {path}: {e}")
                continue
        
        if data is None:
            raise FileNotFoundError("Could not find the Excel file in any of the expected locations")
            
        # Extract the total number of trees from the first row and first column
        total_trees = data.iloc[0, 0]
        
        # Get the data starting from the second row (which should contain lat/long)
        data = data.iloc[1:].reset_index(drop=True)
        
        # Based on the Excel file structure we've seen
        # The third column is actually the zone number (1-4)
        # Rename columns appropriately
        data.columns = ['Latitud', 'Longitud', 'Zona', 'Manzana', 'Nro de Árbol', 'Google Maps', 'Google Earth']
        
        # Convert latitude and longitude to numeric, handling formatting issues
        data['Latitud'] = pd.to_numeric(data['Latitud'], errors='coerce')
        data['Longitud'] = pd.to_numeric(data['Longitud'], errors='coerce')
        
        # Drop rows with missing coordinates
        data = data.dropna(subset=['Latitud', 'Longitud'])
        
        # Fix coordinates - Rivera, Uruguay is around -30.9 latitude and -55.5 longitude
        # Filter coordinates to only include those in Rivera region (within reasonable bounds)
        rivera_lat_min, rivera_lat_max = -31.1, -30.7  # Approximate bounds for Rivera
        rivera_lon_min, rivera_lon_max = -55.7, -55.3  # Approximate bounds for Rivera
        
        # Filter to only include coordinates within Rivera region
        data = data[(data['Latitud'] >= rivera_lat_min) & (data['Latitud'] <= rivera_lat_max) & 
                    (data['Longitud'] >= rivera_lon_min) & (data['Longitud'] <= rivera_lon_max)]
        
        # Format zone values to be "Zona 1", "Zona 2", etc.
        data['Zona'] = data['Zona'].apply(lambda x: f'Zona {int(x)}' if pd.notna(x) and str(x).strip() else 'No especificada')
        
        # Drop rows without a zone assignment
        data = data.dropna(subset=['Zona'])
        
        # Create a species column (not in the original data, so we'll mark as 'No especificada')
        data['Especie'] = 'No especificada'
            
        return data, total_trees
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide empty dataframe with expected structure
        return pd.DataFrame(columns=['Latitud', 'Longitud', 'Especie', 'Zona', 'Zona_1', 'Zona_2', 'Zona_3', 'Zona_4']), 0

# Load the data
data, total_trees = load_data()

# Sidebar filters
st.sidebar.markdown('<div class="sub-header">Filtros</div>', unsafe_allow_html=True)

# Zone filter
st.sidebar.markdown("### Filtrar por Zona")
zones = ['Todos'] + sorted(data['Zona'].unique().tolist())
selected_zone = st.sidebar.selectbox('Seleccionar Zona', zones)

# Species filter if available
if 'Especie' in data.columns and len(data['Especie'].unique()) > 1:
    st.sidebar.markdown("### Filtrar por Especie")
    species = ['Todos'] + sorted(data['Especie'].unique().tolist())
    selected_species = st.sidebar.selectbox('Seleccionar Especie', species)
else:
    selected_species = 'Todos'

# Apply filters
filtered_data = data.copy()
if selected_zone != 'Todos':
    filtered_data = filtered_data[filtered_data['Zona'] == selected_zone]
if selected_species != 'Todos' and 'Especie' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['Especie'] == selected_species]

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="sub-header">Mapa de Ubicación de Árboles</div>', unsafe_allow_html=True)
    
    # Create folium map
    if len(filtered_data) > 0:
        # Calculate the center of the map
        center_lat = filtered_data['Latitud'].mean()
        center_lon = filtered_data['Longitud'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
        
        # Create a marker cluster
        from folium.plugins import MarkerCluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for each tree
        for idx, row in filtered_data.iterrows():
            popup_text = f"""
            <b>Zona:</b> {row['Zona']}<br>
            <b>Especie:</b> {row['Especie']}<br>
            <b>Latitud:</b> {row['Latitud']:.6f}<br>
            <b>Longitud:</b> {row['Longitud']:.6f}
            """
            folium.Marker(
                location=[row['Latitud'], row['Longitud']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='green', icon='tree', prefix='fa')
            ).add_to(marker_cluster)
        
        # Display the map
        folium_static(m, width=800, height=500)
    else:
        st.warning("No hay datos para mostrar en el mapa con los filtros seleccionados.")

with col2:
    st.markdown('<div class="sub-header">Estadísticas</div>', unsafe_allow_html=True)
    
    # Key metrics
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    # Total trees
    st.metric("Total de Árboles en el Dataset", len(data))
    
    # Total trees from first cell if available
    if isinstance(total_trees, (int, float)):
        st.metric("Total de Árboles Registrados", int(total_trees))
    
    # Trees in filtered data
    st.metric("Árboles en Selección Actual", len(filtered_data))
    
    # Distribution by zone
    zone_counts = data['Zona'].value_counts()
    zone_percentages = (zone_counts / zone_counts.sum() * 100).round(1)
    
    for zone, count in zone_counts.items():
        percentage = zone_percentages[zone]
        st.write(f"{zone}: {count} árboles ({percentage}%)")
        st.progress(percentage / 100)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Charts section
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="sub-header">Visualizaciones</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Distribution by zone pie chart
    fig = px.pie(
        names=data['Zona'].value_counts().index,
        values=data['Zona'].value_counts().values,
        title="Distribución de Árboles por Zona",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Species distribution if available
    if 'Especie' in data.columns and len(data['Especie'].unique()) > 1:
        top_species = data['Especie'].value_counts().head(10)
        fig = px.bar(
            x=top_species.index,
            y=top_species.values,
            title="Top 10 Especies más Comunes",
            labels={'x': 'Especie', 'y': 'Cantidad'},
            color=top_species.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Zone distribution by species if available
    if 'Especie' in data.columns and len(data['Especie'].unique()) > 1:
        species_zone = data.groupby(['Zona', 'Especie']).size().reset_index(name='count')
        top_species = data['Especie'].value_counts().nlargest(5).index
        species_zone_filtered = species_zone[species_zone['Especie'].isin(top_species)]
        
        fig = px.bar(
            species_zone_filtered, 
            x='Zona', 
            y='count', 
            color='Especie',
            title="Distribución de Top 5 Especies por Zona",
            labels={'count': 'Cantidad', 'Zona': 'Zona'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Density heatmap if enough data points
    if len(filtered_data) > 10:
        st.subheader("Densidad de Árboles")
        fig = px.density_mapbox(
            filtered_data, 
            lat='Latitud', 
            lon='Longitud', 
            radius=10,
            center=dict(lat=filtered_data['Latitud'].mean(), lon=filtered_data['Longitud'].mean()),
            zoom=13,
            mapbox_style="open-street-map"
        )
        st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="sub-header">Datos Detallados</div>', unsafe_allow_html=True)

# Show sample of filtered data
if len(filtered_data) > 0:
    st.dataframe(
        filtered_data[['Latitud', 'Longitud', 'Especie', 'Zona']].reset_index(drop=True),
        use_container_width=True,
        hide_index=False,
    )
else:
    st.info("No hay datos disponibles con los filtros seleccionados.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 14px;">
    Dashboard creado para visualizar datos de arbolado. © 2025
</div>
""", unsafe_allow_html=True)
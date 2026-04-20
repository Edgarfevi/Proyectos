from __future__ import print_function
from Selector import Selector
from Plotter import Plotter
import ROOT as r

# Evita que se abran ventanas gráficas
r.gROOT.SetBatch(1) 

# 1. Definición de muestras
MCsamples = ['qcd', 'wjets', 'ww', 'wz', 'zz', 'dy', 'single_top', 'ttbar']
plot = Plotter(MCsamples, 'data')

# 2. Asignación de colores
colors = [r.kGray, r.kBlue-1, r.kTeal-1, r.kTeal+1, r.kTeal+4, r.kAzure-8, r.kOrange+1, r.kRed+1]
plot.SetColors(colors)

# 3. Generación de histogramas
print("\n[INFO] Generando histogramas y graficas en ./results...")
plot.SetYtitle('Events')

plot.SetXtitle('p_{T}^{#mu} (GeV)')
plot.Stack('MuonPt')

plot.SetXtitle('N_{jets}')
plot.Stack('NJet')

plot.SetXtitle('N_{b-jets}')
plot.Stack('NbJets')

plot.SetXtitle('E_{T}^{miss} (GeV)')
plot.Stack('MET')

plot.SetXtitle('p_{T}^{#mu} (GeV)')
plot.SetYtitle('Efficiency')
plot.Stack('Eficiencia')

# 4. Impresión de resultados
print("\n--- CONTEOS PARA EL CÁLCULO DE SECCIÓN EFICAZ ---")
plot.PrintCounts('MuonPt')
plot.SaveCounts('MuonPt', 'resultados_finales.txt')

# =============================================================================
# 5. CÁLCULO DE SECCIÓN EFICAZ (Región Fiducial y Eficiencia)
# =============================================================================

n_obs = plot.GetEvents('data', 'MuonPt')
fondos = ['qcd', 'wjets', 'ww', 'wz', 'zz', 'dy', 'single_top']
n_bkg = sum(plot.GetEvents(f, 'MuonPt') for f in fondos)

# Parámetros básicos de la Tabla 1 y Guion
n_gen_ttbar = 36941.0  
lumi = 50.0             
eff_constantes = 0.99 * 0.95 # Eficiencia reco muones * Eficiencia estimada trigger

# --- EXTRACCIÓN DE EVENTOS PUROS (Usando GetEntries para evitar pesos) ---
n_fiduciales = plot.GetHisto('ttbar', 'Fiducial').GetEntries()
n_reconstruidos = plot.GetHisto('ttbar', 'MuonPt').GetEntries()

# --- CÁLCULO SEPARADO ---
if n_fiduciales > 0:
    aceptancia = n_fiduciales / n_gen_ttbar
    eff_reco = n_reconstruidos / n_fiduciales
else:
    print("[WARNING] No se encontraron eventos fiduciales en el archivo. Usando método alternativo para estimar aceptancia y eficiencia.")
    # Fallback matemático por si las variables fiduciales no existen en el archivo
    # (En este caso deshace el peso teórico para estimar la aceptancia real)
    sigma_teo = 165.0
    peso_MC = (lumi * sigma_teo) / n_gen_ttbar
    n_reconstruidos_estimado = plot.GetEvents('ttbar', 'MuonPt') / peso_MC
    
    aceptancia = n_reconstruidos_estimado / n_gen_ttbar
    eff_reco = 1.0 # Absorbida en la aceptancia por falta de datos fiduciales

eff_total = eff_reco * eff_constantes

# --- FÓRMULA FINAL ---
n_neto_data = n_obs - n_bkg

if (lumi * aceptancia * eff_total) > 0:
    sigma_tt = n_neto_data / (lumi * aceptancia * eff_total)
else:
    sigma_tt = 0

print("Aceptancia (A): {:.4f}".format(aceptancia))
# =============================================================================
# 6. Informe final de física
# =============================================================================
with open("results/seccion_eficaz_final.txt", "w") as f_out:
    f_out.write("INFORME FINAL DE MEDICION: QUARK TOP\n")
    f_out.write("="*40 + "\n")
    f_out.write("Eventos Observados (Data):      {:.2f}\n".format(n_obs))
    f_out.write("Eventos de Fondo (Suma MC):     {:.2f}\n".format(n_bkg))
    f_out.write("Eventos Netos de Senal:         {:.2f}\n".format(n_neto_data))
    f_out.write("-" * 40 + "\n")
    if n_fiduciales > 0:
        f_out.write("Eventos Fiduciales (Generador): {:.0f}\n".format(n_fiduciales))
        f_out.write("Eventos Reconstruidos (MC):     {:.0f}\n".format(n_reconstruidos))
    f_out.write("-" * 40 + "\n")
    f_out.write("Aceptancia (A):                 {:.4f}\n".format(aceptancia))
    f_out.write("Eficiencia cinemática (eps_r):  {:.4f}\n".format(eff_reco))
    f_out.write("Eficiencia constante  (eps_c):  {:.4f}\n".format(eff_constantes))
    f_out.write("Eficiencia TOTAL      (eps):    {:.4f}\n".format(eff_total))
    f_out.write("-" * 40 + "\n")
    f_out.write("SECCION EFICAZ FINAL (sigma):   {:.2f} pb\n".format(sigma_tt))
    f_out.write("="*40 + "\n")

print("\n" + "="*45)
print("  ANALISIS COMPLETADO CON EXITO")
print("  Archivo generado: results/seccion_eficaz_final.txt")
print("  Seccion Eficaz Medida: {:.2f} pb".format(sigma_tt))
print("="*45 + "\n")
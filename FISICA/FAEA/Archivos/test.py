from __future__ import print_function
from Plotter import Plotter
import ROOT as r
import math

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
# 5. CÁLCULO DE SECCIÓN EFICAZ Y PROPAGACIÓN DE INCERTIDUMBRES
# =============================================================================

n_obs = plot.GetEvents('data', 'MuonPt')
fondos = ['qcd', 'wjets', 'ww', 'wz', 'zz', 'dy', 'single_top']
n_bkg = sum(plot.GetEvents(f, 'MuonPt') for f in fondos)

# Parámetros básicos 
n_gen_ttbar = 36941.0  
lumi_nom = 50.0             
eff_muon = 0.99
eff_trig = 0.95 # Eficiencia estimada trigger (puedes ajustar a tu valor medido)

# Extracción de eventos
n_fiduciales = plot.GetHisto('ttbar', 'Fiducial').GetEntries()
n_reconstruidos = plot.GetHisto('ttbar', 'MuonPt').GetEntries()

# Cálculo de aceptancia y eficiencia cinemática
if n_fiduciales > 0:
    aceptancia = n_fiduciales / n_gen_ttbar
    eff_reco = n_reconstruidos / n_fiduciales
else:
    # Fallback si no hay eventos fiduciales (por seguridad)
    sigma_teo = 165.0
    peso_MC = (lumi_nom * sigma_teo) / n_gen_ttbar
    aceptancia = (plot.GetEvents('ttbar', 'MuonPt') / peso_MC) / n_gen_ttbar
    eff_reco = 1.0

# Función maestra para calcular sigma iterativamente
def calcular_sigma(obs, bkg, lumi, acc, e_reco, e_muon, e_trig):
    e_tot = e_reco * e_muon * e_trig
    if (lumi * acc * e_tot) > 0:
        return (obs - bkg) / (lumi * acc * e_tot)
    return 0.0

# --- 5.1 CÁLCULO NOMINAL ---
sigma_tt_nom = calcular_sigma(n_obs, n_bkg, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig)

# --- 5.2 INCERTIDUMBRES ESTADÍSTICAS ---
# Asumimos estadística de Poisson para los datos: error = raiz(N)
# Propagación: error_sigma = sigma * (raiz(N_obs) / N_neto)
error_stat = sigma_tt_nom * (math.sqrt(n_obs) / (n_obs - n_bkg)) if (n_obs - n_bkg) > 0 else 0

# --- 5.3 INCERTIDUMBRES SISTEMÁTICAS (NORMALIZACIÓN DE FONDOS) ---
# Extraemos yields individuales
yields = {f: plot.GetEvents(f, 'MuonPt') for f in fondos}
yield_diboson = yields['ww'] + yields['wz'] + yields['zz']

# Variamos cada fondo sumando su error y recalculando sigma
d_wjets = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg + yields['wjets']*0.50, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig))
d_qcd   = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg + yields['qcd']*1.00, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig))
d_dy    = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg + yields['dy']*0.15, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig))
d_stop  = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg + yields['single_top']*0.30, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig))
d_dibos = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg + yield_diboson*0.50, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig))

# --- 5.4 INCERTIDUMBRES SISTEMÁTICAS (EFICIENCIAS) ---
# b-tagging (10% sobre la eficiencia de reconstrucción que lo engloba)
d_btag  = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg, lumi_nom, aceptancia, eff_reco*1.10, eff_muon, eff_trig))

# Muon reco (0.01)
d_muon  = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg, lumi_nom, aceptancia, eff_reco, eff_muon + 0.01, eff_trig))

# Trigger: (1 - eps_tr) / 2
error_trig = (1.0 - eff_trig) / 2.0
d_trig  = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg, lumi_nom, aceptancia, eff_reco, eff_muon, eff_trig + error_trig))

# Suma cuadrática de todas las sistemáticas (sin asumir correlación)
error_syst = math.sqrt(d_wjets**2 + d_qcd**2 + d_dy**2 + d_stop**2 + d_dibos**2 + d_btag**2 + d_muon**2 + d_trig**2)

# --- 5.5 INCERTIDUMBRE DE LUMINOSIDAD ---
d_lumi = abs(sigma_tt_nom - calcular_sigma(n_obs, n_bkg, lumi_nom*1.10, aceptancia, eff_reco, eff_muon, eff_trig))

print(f"Eficiencia reco = {eff_reco:.4f} (variación 10% -> {d_btag:.2f} pb)")
print(f"Eficiencia muon reco = {eff_muon:.4f} (variación 1% -> {d_muon:.2f} pb)")
print(f"Eficiencia trigger = {eff_trig:.4f} (variación 50% -> {d_trig:.2f} pb)")
print(f"Aceptancia = {aceptancia:.4f}")

# =============================================================================
# 6. INFORME FINAL DE FÍSICA
# =============================================================================
with open("results/seccion_eficaz_final.txt", "w") as f_out:
    f_out.write("INFORME FINAL DE MEDICION: QUARK TOP\n")
    f_out.write("="*50 + "\n")
    f_out.write("Eventos Observados (Data):      {:.2f}\n".format(n_obs))
    f_out.write("Eventos de Fondo (Suma MC):     {:.2f}\n".format(n_bkg))
    f_out.write("Eventos Netos de Senal:         {:.2f}\n".format(n_obs - n_bkg))
    f_out.write("-" * 50 + "\n")
    f_out.write("Aceptancia (A):                 {:.4f}\n".format(aceptancia))
    f_out.write("Eficiencia TOTAL (eps):         {:.4f}\n".format(eff_reco * eff_muon * eff_trig))
    f_out.write("-" * 50 + "\n")
    f_out.write("SECCION EFICAZ FINAL (sigma):   {:.2f} pb\n".format(sigma_tt_nom))
    f_out.write("-" * 50 + "\n")
    f_out.write("DESGLOSE DE INCERTIDUMBRES:\n")
    f_out.write("  Estadistica:    +/- {:.2f} pb\n".format(error_stat))
    f_out.write("  Sistematica:    +/- {:.2f} pb\n".format(error_syst))
    f_out.write("  Luminosidad:    +/- {:.2f} pb\n".format(d_lumi))
    f_out.write("="*50 + "\n")
    f_out.write("RESULTADO FINAL:\n")
    f_out.write("sigma = {:.2f} +/- {:.2f} (stat) +/- {:.2f} (syst) +/- {:.2f} (lumi) pb\n".format(
        sigma_tt_nom, error_stat, error_syst, d_lumi))
    f_out.write("="*50 + "\n")

print("\n" + "="*55)
print("  ANALISIS COMPLETADO CON EXITO")
print("  Archivo generado: results/seccion_eficaz_final.txt")
print("  Seccion Eficaz Medida: {:.2f} +/- {:.2f} (stat) +/- {:.2f} (syst) +/- {:.2f} (lumi) pb".format(
    sigma_tt_nom, error_stat, error_syst, d_lumi))
print("="*55 + "\n")
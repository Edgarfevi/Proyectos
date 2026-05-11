from __future__ import print_function
import os
import ROOT as r
import numpy as np

class Selector:
    def __init__(self, filename=''):
        self.name = filename
        self.histograms = [] 
        self.filename = self.name if self.name.endswith('.root') else self.name + '.root'
        
        if not os.path.exists(self.filename): 
            self.filename = 'Datos/' + self.filename
        
        self.CreateHistograms()
        self.Loop()
        self.Eficiencia()

    def CreateHistograms(self):
        self.histograms = []
        # Histograma para la Region Fiducial (Nivel Generacion - Eq. 5 del guion)
        self.histograms.append(r.TH1F(self.name + '_Fiducial', 'Region Fiducial;p_{T}^{gen} (GeV);Events', 20, 20, 200))
        
        # Histogramas para Eficiencia de Trigger (Eq. 4 del guion)
        self.histograms.append(r.TH1F(self.name + '_MuonPt_st', ';p_{T}^{#mu} (GeV);Events', 20, 20, 50))   
        self.histograms.append(r.TH1F(self.name + '_MuonPt_pass', ';p_{T}^{#mu} (GeV);Events', 20, 20, 50)) 
        self.histograms.append(r.TH1F(self.name + '_Eficiencia', ';p_{T}^{#mu} (GeV);Eficiencia', 20, 20, 50))
        
        # Histogramas de Analisis (Nivel Detector - Eq. 1 del guion)
        self.histograms.append(r.TH1F(self.name + '_MuonPt', ';p_{T}^{#mu} (GeV);Events', 20, 20, 200))
        self.histograms.append(r.TH1F(self.name + '_JetPt', ';p_{T}^{jet} (GeV);Events', 20, 0, 200))
        self.histograms.append(r.TH1F(self.name + '_NJet', ';N_{jets};Events', 10, 0, 10))
        self.histograms.append(r.TH1F(self.name + '_NbJets', ';N_{b-jets};Events', 5, 0, 5))
        self.histograms.append(r.TH1F(self.name + '_MET', ';E_{T}^{miss} (GeV);Events', 20, 0, 200))

    def GetHisto(self, name):
        target = self.name + '_' + name
        for h in self.histograms:
            if target == h.GetName(): 
                return h
        return r.TH1F()

    def Loop(self):
        f = r.TFile.Open(self.filename)
        tree = f.Get("events")
        
        # Contadores para Eficiencia b-tagging y Aceptancia
        total_b = 0
        total_correct_tag = 0
        self.total_gen = 0
        self.fiducial_events = 0
        
        for event in tree:
            # Peso de evento
            weight = event.EventWeight if self.name != 'data' else 1.0

            # --- 0. CALCULO DE LA REGION FIDUCIAL Y ACEPTANCIA ---
            if self.name == 'ttbar':
                # Sumamos todos los eventos al denominador de la aceptancia
                self.total_gen += weight

                # 1. Definimos los vectores con su masa para extraer Eta() y Pt() correctamente
                mc_muon = r.TLorentzVector()
                mc_muon.SetPxPyPzE(event.MClepton_px, event.MClepton_py, event.MClepton_pz, 0)

                mc_b_had = r.TLorentzVector()
                mc_b_had.SetPxPyPzE(event.MChadronicBottom_px, event.MChadronicBottom_py, event.MChadronicBottom_pz, np.sqrt(event.MChadronicBottom_px**2 + event.MChadronicBottom_py**2 + event.MChadronicBottom_pz**2+4.18**2))

                mc_b_lep = r.TLorentzVector()
                mc_b_lep.SetPxPyPzE(event.MCleptonicBottom_px, event.MCleptonicBottom_py, event.MCleptonicBottom_pz, np.sqrt(event.MCleptonicBottom_px**2 + event.MCleptonicBottom_py**2 + event.MCleptonicBottom_pz**2+4.18**2))

                mc_q1_W = r.TLorentzVector()
                mc_q1_W.SetPxPyPzE(event.MChadronicWDecayQuark_px, event.MChadronicWDecayQuark_py, event.MChadronicWDecayQuark_pz, np.sqrt(event.MChadronicWDecayQuark_px**2 + event.MChadronicWDecayQuark_py**2 + event.MChadronicWDecayQuark_pz**2+0.03**2))

                mc_q2_W = r.TLorentzVector()
                mc_q2_W.SetPxPyPzE(event.MChadronicWDecayQuarkBar_px, event.MChadronicWDecayQuarkBar_py, event.MChadronicWDecayQuarkBar_pz, np.sqrt(event.MChadronicWDecayQuarkBar_px**2 + event.MChadronicWDecayQuarkBar_py**2 + event.MChadronicWDecayQuarkBar_pz**2+0.03**2))

                # 2. Contamos quarks visibles exigiendo pT > 40 Y |eta| < 2.4
                quarks = [mc_b_had, mc_b_lep, mc_q1_W, mc_q2_W]
                n_quarks_visibles = sum(1 for q in quarks if q.Pt() > 30 and abs(q.Eta()) < 2.4)

                # 3. APLICAMOS CORTES FIDUCIALES (Análogos a la señal)
                met_gen = np.sqrt(event.MCneutrino_px**2 + event.MCneutrino_py**2)                
                if n_quarks_visibles >= 3 and mc_muon.Pt() > 28 and abs(mc_muon.Eta()) < 2.4 and met_gen > 20 and abs(event.MCleptonPDGid) == 13:
                    # Llenamos el histograma de la región fiducial con el peso original
                    self.GetHisto('Fiducial').Fill(mc_b_had.Pt(), weight)
                    # Sumamos al numerador de la aceptancia
                    self.fiducial_events += weight

                # 4. Cálculo para Eficiencia de b-tagging a nivel MC
                for i in range(event.NJet):
                    jet = r.TLorentzVector()
                    jet.SetPxPyPzE(event.Jet_Px[i], event.Jet_Py[i], event.Jet_Pz[i], event.Jet_E[i])
                    
                    # Usamos DeltaR nativo de ROOT
                    delta_R_had = mc_b_had.DeltaR(jet)
                    delta_R_lep = mc_b_lep.DeltaR(jet)
                    
                    if delta_R_had < 0.3 or delta_R_lep < 0.3:
                        total_b += 1
                        if event.Jet_btag[i] > 2.0:
                            total_correct_tag += 1

            # --- 1. SELECCION DE RECONSTRUCCION (NIVEL DETECTOR) ---
            if event.NMuon != 1: 
                continue 
            muon = r.TLorentzVector()
            muon.SetPxPyPzE(event.Muon_Px[0], event.Muon_Py[0], event.Muon_Pz[0], event.Muon_E[0])
            
            # Eficiencia de Trigger (Denominador)
            if muon.Pt() > 20: 
                self.GetHisto('MuonPt_st').Fill(muon.Pt(), weight)

            # Requisito de Trigger (Si no paso, no se guarda)
            if not event.triggerIsoMu24: 
                continue
            
            # Eficiencia de Trigger (Numerador)
            if muon.Pt() > 20: 
                self.GetHisto('MuonPt_pass').Fill(muon.Pt(), weight)

            # --- CORTES DE SELECCION FINAL (REGION DE SEÑAL) ---
            # Exigimos también el Eta al muon para que coincida con la región fiducial
            if muon.Pt() < 28 or abs(muon.Eta()) > 2.4: 
                continue

            met_val = np.sqrt(event.MET_px**2 + event.MET_py**2)
            if met_val < 20:
                continue

            # Contadores y lista para almacenar el pT de los jets válidos
            n_valid_jets = 0
            n_btags = 0
            valid_jet_pts = []

            for i in range(event.NJet):
                jet = r.TLorentzVector()
                jet.SetPxPyPzE(event.Jet_Px[i], event.Jet_Py[i], event.Jet_Pz[i], event.Jet_E[i])
                
                # Si el jet está fuera de la región o no tiene pT suficiente, lo ignoramos
                if abs(jet.Eta()) > 2.4 or jet.Pt() < 30: 
                    continue 
                
                # Si llegó aquí, el jet es válido
                n_valid_jets += 1
                valid_jet_pts.append(jet.Pt())
                
                # Comprobamos si este jet válido es un b-jet
                if event.Jet_btag[i] > 2.0:
                    n_btags += 1

            # AHORA aplicamos los cortes sobre los jets que sí pasaron la geometría
            if n_valid_jets < 3: 
                continue
            if n_btags < 1:
                continue
            
            # Correcciones del Scale Factor de b-tagging (solo para MC)
            if self.name != 'data':
                if n_btags == 1:
                    weight *= 0.9
                elif n_btags >= 2:
                    weight *= 0.86

            # Relleno de Histogramas Finales (usando n_valid_jets y la lista de pT filtrada)
            for pt in valid_jet_pts:
                self.GetHisto('JetPt').Fill(pt, weight)
                
            self.GetHisto('MuonPt').Fill(muon.Pt(), weight)
            self.GetHisto('NJet').Fill(n_valid_jets, weight)
            self.GetHisto('NbJets').Fill(n_btags, weight)
            self.GetHisto('MET').Fill(met_val, weight)

        # Resultados Finales del Loop
        if self.name == 'ttbar':
            self.eficiencia_b = total_correct_tag / total_b * 0.9 if total_b > 0 else 0.0
            print(f"Eficiencia b-tagging (nivel generación): {self.eficiencia_b:.4f}")
            
            # Cálculo de la Aceptancia
            if self.total_gen > 0:
                self.aceptancia = self.fiducial_events / self.total_gen
                print(f"Aceptancia (A): {self.aceptancia:.4f}")
                
        f.Close()

    def Eficiencia(self):
        self.GetHisto('Eficiencia').Divide(self.GetHisto('MuonPt_pass'), self.GetHisto('MuonPt_st'), 1.0, 1.0, "B")
        
        if self.name == 'ttbar':
            h_pass = self.GetHisto('MuonPt_pass')
            h_st = self.GetHisto('MuonPt_st')

            # Buscamos el bin que corresponde a 28 GeV (tu nuevo corte de análisis)
            bin_40 = h_pass.FindBin(28.0)  # Usamos 28 GeV para ser consistentes con el corte de MET
            end_bin = h_pass.GetNbinsX() + 1

            # Obtenemos la eficiencia usando la Integral (eventos reales / totales)
            eventos_pass = h_pass.Integral(bin_40, end_bin)
            eventos_totales = h_st.Integral(bin_40, end_bin)

            if eventos_totales > 0:
                self.eficiencia_trigger = eventos_pass / eventos_totales
                print(f"Eficiencia del trigger para pT > 28 GeV: {self.eficiencia_trigger:.4f}")
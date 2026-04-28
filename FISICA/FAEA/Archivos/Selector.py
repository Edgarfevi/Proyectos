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
        for event in tree:
            # --- 0. CALCULO DE LA REGION FIDUCIAL COMPLETO ---
            if self.name == 'ttbar':
                # 1. Calculamos pT de los dos b-quarks
                pt_b_had = np.sqrt(event.MChadronicBottom_px**2 + event.MChadronicBottom_py**2)
                pt_b_lep = np.sqrt(event.MCleptonicBottom_px**2 + event.MCleptonicBottom_py**2)
                
                # 2. Calculamos pT de los quarks del W (los que darán los otros jets)
                pt_q1_W = np.sqrt(event.MChadronicWDecayQuark_px**2 + event.MChadronicWDecayQuark_py**2)
                pt_q2_W = np.sqrt(event.MChadronicWDecayQuarkBar_px**2 + event.MChadronicWDecayQuarkBar_py**2)
                
                # 3. Calculamos pT del lepton (muon) para el corte de eficiencia
                lepton_pt = np.sqrt(event.MClepton_px**2 + event.MClepton_py**2)
                
                # Definimos la REGION FIDUCIAL:
                # Pedimos que los 4 quarks principales tengan pT > 30 GeV 
                # (Esto asegura que el evento "debería" dar al menos 4 jets en el detector)
                if pt_b_had > 30 and pt_b_lep > 30 and pt_q1_W > 30 and pt_q2_W > 30 and lepton_pt > 27:
                    self.GetHisto('Fiducial').Fill(pt_b_had)

            # --- 1. SELECCION DE RECONSTRUCCION (NIVEL DETECTOR) ---
            if event.NMuon != 1: 
                continue 
            muon = r.TLorentzVector()
            muon.SetPxPyPzE(event.Muon_Px[0], event.Muon_Py[0], event.Muon_Pz[0], event.Muon_E[0])
            
            # Peso de evento (Luminosidad y Seccion Eficaz Teorica ya incluidas en EventWeight) [cite: 78, 102]
            weight = event.EventWeight if self.name != 'data' else 1.0

            # Eficiencia de Trigger (Denominador)
            if muon.Pt() > 20: 
                self.GetHisto('MuonPt_st').Fill(muon.Pt(), weight)

            # Requisito de Trigger (Si no paso, no se guarda) [cite: 177, 191]
            if not event.triggerIsoMu24: 
                continue
            
            # Eficiencia de Trigger (Numerador)
            if muon.Pt() > 20: 
                self.GetHisto('MuonPt_pass').Fill(muon.Pt(), weight)

            # --- CORTES DE SELECCION FINAL (REGION DE SEÑAL) ---
            if muon.Pt() < 27: 
                continue
            if event.NJet < 3: 
                continue 
            
            # B-Tagging (Eq. de b-tagging del guion)
            nbtags = sum(1 for i in range(event.NJet) if event.Jet_btag[i] > 2.0)
            if nbtags < 2: 
                continue 
            
            # Factor de escala b-tagging para MC [cite: 209]
            if self.name != 'data': 
                weight *= 0.9 

            # Relleno de Histogramas Finales
            self.GetHisto('MuonPt').Fill(muon.Pt(), weight)
            self.GetHisto('NJet').Fill(event.NJet, weight)
            self.GetHisto('NbJets').Fill(nbtags, weight)
            self.GetHisto('MET').Fill(np.sqrt(event.MET_px**2 + event.MET_py**2), weight)
        f.Close()


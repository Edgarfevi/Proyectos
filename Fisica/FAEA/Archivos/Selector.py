from __future__ import print_function

import os

import ROOT as r
import numpy as np


class Selector:
    """Procesa un fichero ROOT y construye los histogramas de análisis.

    La clase ejecuta tres pasos principales al instanciarse:
    1. crear los histogramas necesarios,
    2. recorrer el árbol de eventos,
    3. calcular eficiencias globales.
    """

    def __init__(self, filename=""):
        self.name = filename
        self.histograms = []
        self.filename = self.name if self.name.endswith(".root") else self.name + ".root"

        if not os.path.exists(self.filename):
            self.filename = os.path.join("Datos", self.filename)

        self.CreateHistograms()
        self.Loop()
        self.Eficiencia()

    def CreateHistograms(self):
        """Define todos los histogramas usados en el análisis."""

        self.histograms = []

        # Región fiducial a nivel de generación.
        self.histograms.append(
            r.TH1F(self.name + "_Fiducial", "Region Fiducial;p_{T}^{gen} (GeV);Events", 20, 20, 200)
        )

        # Histogramas para la eficiencia del trigger.
        self.histograms.append(r.TH1F(self.name + "_MuonPt_st", ";p_{T}^{#mu} (GeV);Events", 20, 20, 50))
        self.histograms.append(r.TH1F(self.name + "_MuonPt_pass", ";p_{T}^{#mu} (GeV);Events", 20, 20, 50))
        self.histograms.append(r.TH1F(self.name + "_Eficiencia", ";p_{T}^{#mu} (GeV);Eficiencia", 20, 20, 50))

        # Distribuciones finales a nivel detector.
        self.histograms.append(r.TH1F(self.name + "_MuonPt", ";p_{T}^{#mu} (GeV);Events", 20, 20, 200))
        self.histograms.append(r.TH1F(self.name + "_JetPt", ";p_{T}^{jet} (GeV);Events", 20, 0, 200))
        self.histograms.append(r.TH1F(self.name + "_NJet", ";N_{jets};Events", 10, 0, 10))
        self.histograms.append(r.TH1F(self.name + "_NbJets", ";N_{b-jets};Events", 5, 0, 5))
        self.histograms.append(r.TH1F(self.name + "_MET", ";E_{T}^{miss} (GeV);Events", 20, 0, 200))
        self.histograms.append(r.TH1F(self.name + "_MuonEta", ";#eta^{#mu};Events", 20, -2.4, 2.4))

    def GetHisto(self, name):
        """Devuelve el histograma con nombre interno `self.name + '_' + name`."""

        target = self.name + "_" + name
        for hist in self.histograms:
            if target == hist.GetName():
                return hist
        return r.TH1F()

    def Loop(self):
        """Recorre el árbol de eventos y llena histogramas y contadores."""

        root_file = r.TFile.Open(self.filename)
        tree = root_file.Get("events")

        # Contadores auxiliares para la b-tagging y la aceptancia.
        total_b = 0
        total_correct_tag = 0
        self.total_gen = 0
        self.fiducial_events = 0

        for event in tree:
            # Peso nominal del evento; en datos se fuerza a 1.
            weight = event.EventWeight if self.name != "data" else 1.0

            # --------------------------------------------------------------
            # 0. Región fiducial y aceptancia (solo para ttbar MC)
            # --------------------------------------------------------------
            if self.name == "ttbar":
                # Denominador de la aceptancia.
                self.total_gen += weight

                # Vectores de verdad para reconstruir pT y eta de los productos.
                mc_muon = r.TLorentzVector()
                mc_muon.SetPxPyPzE(event.MClepton_px, event.MClepton_py, event.MClepton_pz, 0)

                mc_b_had = r.TLorentzVector()
                mc_b_had.SetPxPyPzE(
                    event.MChadronicBottom_px,
                    event.MChadronicBottom_py,
                    event.MChadronicBottom_pz,
                    np.sqrt(
                        event.MChadronicBottom_px**2
                        + event.MChadronicBottom_py**2
                        + event.MChadronicBottom_pz**2
                        + 4.18**2
                    ),
                )

                mc_b_lep = r.TLorentzVector()
                mc_b_lep.SetPxPyPzE(
                    event.MCleptonicBottom_px,
                    event.MCleptonicBottom_py,
                    event.MCleptonicBottom_pz,
                    np.sqrt(
                        event.MCleptonicBottom_px**2
                        + event.MCleptonicBottom_py**2
                        + event.MCleptonicBottom_pz**2
                        + 4.18**2
                    ),
                )

                mc_q1_W = r.TLorentzVector()
                mc_q1_W.SetPxPyPzE(
                    event.MChadronicWDecayQuark_px,
                    event.MChadronicWDecayQuark_py,
                    event.MChadronicWDecayQuark_pz,
                    np.sqrt(
                        event.MChadronicWDecayQuark_px**2
                        + event.MChadronicWDecayQuark_py**2
                        + event.MChadronicWDecayQuark_pz**2
                        + 0.03**2
                    ),
                )

                mc_q2_W = r.TLorentzVector()
                mc_q2_W.SetPxPyPzE(
                    event.MChadronicWDecayQuarkBar_px,
                    event.MChadronicWDecayQuarkBar_py,
                    event.MChadronicWDecayQuarkBar_pz,
                    np.sqrt(
                        event.MChadronicWDecayQuarkBar_px**2
                        + event.MChadronicWDecayQuarkBar_py**2
                        + event.MChadronicWDecayQuarkBar_pz**2
                        + 0.03**2
                    ),
                )

                # Contamos quarks visibles para definir la región fiducial.
                quarks = [mc_b_had, mc_b_lep, mc_q1_W, mc_q2_W]
                n_quarks_visibles = sum(1 for quark in quarks if quark.Pt() > 30 and abs(quark.Eta()) < 2.4)

                # Corte fiducial aplicado sobre la verdad MC.
                met_gen = np.sqrt(event.MCneutrino_px**2 + event.MCneutrino_py**2)
                if (
                    n_quarks_visibles >= 3
                    and mc_muon.Pt() > 27
                    and abs(mc_muon.Eta()) < 2.4
                    and abs(event.MCleptonPDGid) == 13
                    and met_gen > 20
                ):
                    self.GetHisto("Fiducial").Fill(mc_b_had.Pt(), weight)
                    self.fiducial_events += weight

                # Eficiencia de b-tagging a nivel de generación.
                for i in range(event.NJet):
                    jet = r.TLorentzVector()
                    jet.SetPxPyPzE(event.Jet_Px[i], event.Jet_Py[i], event.Jet_Pz[i], event.Jet_E[i])

                    delta_R_had = mc_b_had.DeltaR(jet)
                    delta_R_lep = mc_b_lep.DeltaR(jet)

                    if delta_R_had < 0.3 or delta_R_lep < 0.3:
                        total_b += 1
                        if event.Jet_btag[i] > 2.0:
                            total_correct_tag += 1

            # --------------------------------------------------------------
            # 1. Selección de reconstrucción a nivel detector
            # --------------------------------------------------------------
            if event.NMuon != 1 or event.NElectron != 0:
                continue

            muon = r.TLorentzVector()
            muon.SetPxPyPzE(event.Muon_Px[0], event.Muon_Py[0], event.Muon_Pz[0], event.Muon_E[0])

            # Denominador de la eficiencia del trigger.
            if muon.Pt() > 20:
                self.GetHisto("MuonPt_st").Fill(muon.Pt(), weight)

            # Si no pasa el trigger, el evento no se conserva.
            if not event.triggerIsoMu24:
                continue

            # Numerador de la eficiencia del trigger.
            if muon.Pt() > 20:
                self.GetHisto("MuonPt_pass").Fill(muon.Pt(), weight)

            # --------------------------------------------------------------
            # 2. Selección final de la región de señal
            # --------------------------------------------------------------
            if muon.Pt() < 27 or abs(muon.Eta()) > 2.4:
                continue

            met_val = np.sqrt(event.MET_px**2 + event.MET_py**2)
            if met_val < 20:
                continue

            # Variables de control para los jets seleccionados.
            n_valid_jets = 0
            n_btags = 0
            valid_jet_pts = []

            for i in range(event.NJet):
                jet = r.TLorentzVector()
                jet.SetPxPyPzE(event.Jet_Px[i], event.Jet_Py[i], event.Jet_Pz[i], event.Jet_E[i])

                # Solo aceptamos jets dentro de la región cinemática de interés.
                if abs(jet.Eta()) > 2.4 or jet.Pt() < 30:
                    continue

                n_valid_jets += 1
                valid_jet_pts.append(jet.Pt())

                # Etiquetado b-jet sobre jets ya aceptados.
                if event.Jet_btag[i] > 2.0:
                    n_btags += 1

            if n_valid_jets < 3:
                continue
            if n_btags < 1:
                continue

            # Corrección del scale factor de b-tagging para MC.
            if self.name != "data":
                if n_btags == 1:
                    weight *= 0.9
                elif n_btags >= 2:
                    weight *= 0.86

            # Llenado de histogramas finales.
            for pt in valid_jet_pts:
                self.GetHisto("JetPt").Fill(pt, weight)

            self.GetHisto("MuonPt").Fill(muon.Pt(), weight)
            self.GetHisto("NJet").Fill(n_valid_jets, weight)
            self.GetHisto("NbJets").Fill(n_btags, weight)
            self.GetHisto("MET").Fill(met_val, weight)
            self.GetHisto("MuonEta").Fill(muon.Eta(), weight)

        # Resultados finales del recorrido.
        if self.name == "ttbar":
            self.eficiencia_b = total_correct_tag / total_b * 0.9 if total_b > 0 else 0.0
            print(f"Eficiencia b-tagging (nivel generación): {self.eficiencia_b:.4f}")

            if self.total_gen > 0:
                self.aceptancia = self.fiducial_events / self.total_gen
                print(f"Aceptancia (A): {self.aceptancia:.4f}")

        root_file.Close()

    def Eficiencia(self):
        """Calcula la eficiencia del trigger a partir de los histogramas rellenos."""

        self.GetHisto("Eficiencia").Divide(
            self.GetHisto("MuonPt_pass"),
            self.GetHisto("MuonPt_st"),
            1.0,
            1.0,
            "B",
        )

        if self.name == "ttbar":
            h_pass = self.GetHisto("MuonPt_pass")
            h_st = self.GetHisto("MuonPt_st")

            # Bin asociado al corte pT > 27 GeV.
            bin_27 = h_pass.FindBin(27.0)
            end_bin = h_pass.GetNbinsX() + 1

            # Eficiencia integrada a partir del umbral elegido.
            eventos_pass = h_pass.Integral(bin_27, end_bin)
            eventos_totales = h_st.Integral(bin_27, end_bin)

            if eventos_totales > 0:
                self.eficiencia_trigger = eventos_pass / eventos_totales
                print(f"Eficiencia del trigger para pT > 27 GeV: {self.eficiencia_trigger:.4f}")
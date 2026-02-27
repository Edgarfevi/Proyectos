from __future__ import print_function
import os
import warnings as wr
import ROOT as r

class Selector:
    ''' Class to do an event selection'''
    ### =============================================
    ### Constructor
    def __init__(self, filename = ''):
        ''' Initialize a new Selector by giving the name of a sample.root file '''
        self.name = filename
        self.filename = self.name
        if self.filename[-5:] != '.root': self.filename += '.root'
        if not os.path.exists(self.filename): self.filename = '/home/edgar/GitHub/Proyectos/Proyectos_python/FISICA/FAEA/Datos/' + self.filename
        if not os.path.exists(self.filename):
            if (self.name != ''): wr.warn("[Selector::constructor] WARNING: file {f} not found".format(f = self.name))
        else:
            self.CreateHistograms()
            self.Loop()
        return

    ### =============================================
    ### Attributes
    # General
    histograms = []
    name = ""
    filename = ""

    # Histogram variables
    muon1_pt = -99
    invmass = -99
    weight = -99


    ### =============================================
    ### Methods
    def CreateHistograms(self):
        ''' CREATE YOUR HISTOGRAMS HERE '''
        self.histograms = []
        self.histograms.append(r.TH1F(self.name + '_MuonPt',     ';p_{T}^{#mu} (GeV);Events', 20, 0, 200))
        self.histograms.append(r.TH1F(self.name + '_DiMuonMass', ';m^{#mu#mu} (GeV);Events',  20, 0, 200))
        return


    def GetHisto(self, name):
        ''' Use this method to access to any stored histogram '''
        for h in self.histograms:
            n = h.GetName()
            if self.name + '_' + name == n: return h
        wr.warn("[Selector::GetHisto] WARNING: histogram {h} not found.".format(h = name))
        return r.TH1F()


    def FillHistograms(self):
        self.GetHisto('MuonPt').Fill(self.muon1_pt,    self.weight)
        self.GetHisto('DiMuonMass').Fill(self.invmass, self.weight)
        return


    def Loop(self):
        ''' Main method, used to loop over all the entries '''
        f = r.TFile.Open(self.filename)
        tree = f.events

        nEvents = tree.GetEntries()

        print("Opening file {f} and looping over {n} events...".format(f = self.filename, n = nEvents))
        for event in tree:
            ### Selection
            if not event.triggerIsoMu24: continue # Events must pass the trigger
            if event.NMuon != 2: continue         # Selecting events with 2 muons

            ### Variable calculation
            muon1 = r.TLorentzVector()
            muon1.SetPxPyPzE(event.Muon_Px[0], event.Muon_Py[0], event.Muon_Pz[0], event.Muon_E[0])
            muon2 = r.TLorentzVector()
            muon2.SetPxPyPzE(event.Muon_Px[1], event.Muon_Py[1], event.Muon_Pz[1], event.Muon_E[1])

            self.muon1_pt = muon1.Pt()
            self.invmass  = (muon1 + muon2).M()
            self.weight   = event.EventWeight if not self.name == 'data' else 1

            ### Filling
            self.FillHistograms()
        return

from __future__ import print_function
import warnings as wr
import os
from Selector import Selector
import ROOT as r

class Plotter:
    def __init__(self, backgrounds, data='', path="./results"):
        self.data = data
        self.backgrounds = backgrounds
        self.savepath = path
        
        self.listOfSelectors = [] 
        self.colors = []
        self.xtitle = ''
        self.ytitle = ''
        
        counter = 0
        for p in self.backgrounds:
            self.listOfSelectors.append(Selector(p))
            self.colors.append(counter+1)
            counter += 1

        self.dataSelector = Selector(self.data) if self.data != '' else None

    def SetColors(self, col): self.colors = col
    def SetXtitle(self, tit): self.xtitle = tit
    def SetYtitle(self, tit): self.ytitle = tit

    def GetHisto(self, process, name):
        if process == self.data and self.dataSelector:
            return self.dataSelector.GetHisto(name)
        for s in self.listOfSelectors:
            if process == s.name: 
                return s.GetHisto(name)
        wr.warn("[Plotter::GetHisto] WARNING: histogram {} for process {} not found!".format(name, process))
        return r.TH1F()

    def GetEvents(self, process, name):
        return self.GetHisto(process, name).Integral()

    def Stack(self, name):
        if not os.path.exists(self.savepath): 
            os.makedirs(self.savepath)
        
        # --- EXCEPCIÓN PARA LA GRÁFICA DE EFICIENCIA ---
        if "Eficiencia" in name:
            c = r.TCanvas('c_'+name, '', 800, 600)
            
            # EL GUION PIDE USAR LA SIMULACIÓN DE TTBAR
            h_eff = self.GetHisto('ttbar', name)
            h_eff.SetMarkerStyle(20)
            h_eff.SetMarkerColor(r.kRed+1)
            h_eff.SetLineColor(r.kRed+1)
            h_eff.SetTitle("Eficiencia del Trigger (MC ttbar);p_{T}^{#mu} (GeV);Eficiencia")
            h_eff.SetMaximum(1.2)
            h_eff.SetMinimum(0.0)
            h_eff.Draw("PE") 
                
            c.Print(self.savepath + "/" + name + ".png")
            c.Close()
            return
        # ----------------------------------------------
        
        c = r.TCanvas('c_'+name, '', 800, 600)
        hstack = r.THStack('hstack_' + name, "")
        l = r.TLegend(0.75, 0.6, 0.89, 0.89)
        l.SetBorderSize(0)
        
        for i, s in enumerate(self.listOfSelectors):
            h = s.GetHisto(name)
            h.SetFillColor(self.colors[i])
            h.SetLineColor(1)
            hstack.Add(h)
            l.AddEntry(h, s.name, "f")
        
        hstack.SetMaximum(hstack.GetMaximum() * 1.2)
        hstack.Draw("hist")
        
        if self.xtitle != '': 
            hstack.GetXaxis().SetTitle(self.xtitle)
        if self.ytitle != '': 
            hstack.GetYaxis().SetTitle(self.ytitle)
        
        if self.dataSelector:
            hdata = self.dataSelector.GetHisto(name)
            hdata.SetMarkerStyle(20)
            hdata.Draw("pesame")
            l.AddEntry(hdata, "data", "p")
        
        l.Draw()
        c.Print(self.savepath + "/" + name + ".png")
        c.Close()

    def PrintCounts(self, name):
        print("\nPrinting events for {}:".format(name))
        for s in self.listOfSelectors:
            print("{}: {:.2f}".format(s.name, s.GetHisto(name).Integral()))
        if self.dataSelector:
            print("data: {:.2f}".format(self.dataSelector.GetHisto(name).Integral()))

    def SaveCounts(self, name, overridename=""):
        filename = overridename if overridename else "yields_" + name
        if ".txt" not in filename: 
            filename += ".txt"
        if not os.path.exists(self.savepath): 
            os.makedirs(self.savepath)
        
        yields_mc = {}
        total_bkg, n_signal = 0.0, 0.0
        
        for s in self.listOfSelectors:
            val = s.GetHisto(name).Integral()
            yields_mc[s.name] = val
            if s.name == 'ttbar': 
                n_signal = val
            else: 
                total_bkg += val

        n_obs = self.dataSelector.GetHisto(name).Integral() if self.dataSelector else 0.0

        if self.listOfSelectors[7].eficiencia_b is not None:
            self.eff_b = self.listOfSelectors[7].eficiencia_b

        with open(self.savepath + "/" + filename, "w") as f:
            f.write("====================================================\n")
            f.write("      INFORME DE EVENTOS: {:<20}\n".format(name))
            f.write("====================================================\n\n")
            f.write("{:<20} | {:>15}\n".format("PROCESO", "EVENTOS"))
            f.write("-" * 40 + "\n")
            
            for proc, val in yields_mc.items():
                if proc != 'ttbar':
                    f.write("{:<20} | {:>15.2f}\n".format(proc, val))
            
            f.write("-" * 40 + "\n")
            f.write("{:<20} | {:>15.2f}\n".format("TOTAL FONDO (MC)", total_bkg))
            f.write("{:<20} | {:>15.2f}\n".format("SENAL EXPECT. (tt)", n_signal))
            f.write("-" * 40 + "\n")
            f.write("{:<20} | {:>15.2f}\n".format("OBSERVADO (DATA)", n_obs))
            f.write("====================================================\n\n")
            
            if total_bkg + n_signal > 0:
                f.write("Pureza esperada (S/S+B): {:.2f}%\n".format(n_signal / (total_bkg + n_signal) * 100))
            f.write("Exceso de datos (S_neta): {:.2f} eventos\n".format(n_obs - total_bkg))
from __future__ import print_function
from Selector import Selector
from Plotter import Plotter
import ROOT as r

r.gROOT.SetBatch(1) # To work only by batch (i.e. through terminal, w/o windows)

# Create a selector for ttbar sample
#mySelector = Selector('ttbar')
# Obtain the histogram of invariant mass of the muons from the selector
#h = mySelector.GetHisto('InvMass')
# Do whatever with it...
#print 'The histogram InvMass for the ttbar sample has %i entries' %h.GetEntries()


# Create an object plotter thar contains all MC samples and uses data:
MCsamples = ['qcd', 'wjets', 'ww', 'wz', 'zz', 'dy', 'single_top', 'ttbar']
plot = Plotter(MCsamples, 'data')

# Set colors for each process... We can use colors defined in ROOT
colors = [r.kGray, r.kBlue-1, r.kTeal-1, r.kTeal+1, r.kTeal+4, r.kAzure-8, r.kOrange+1, r.kRed+1]
plot.SetColors(colors)

# Set other plotting options (Legend position, size, titles, etc...)
plot.SetXtitle('p_{T}^{#mu} (GeV)')
plot.SetYtitle('Events')
plot.SetTitle('')

# Draw the stack plot for data and simulation
plot.Stack('MuonPt')

plot.SetXtitle('m_{#mu#mu} (GeV)')
plot.Stack('DiMuonMass')

# Print the contributions for each background and observed data
plot.PrintCounts('MuonPt')
plot.SaveCounts('MuonPt')

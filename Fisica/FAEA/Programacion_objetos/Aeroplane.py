class  Aeroplane :
    def  SetPassengers (self,  nPassengers ,  meanWeight = 65) :
        if nPassengers > self.capacity:
            print('\nNo caben tantos pasajeros!!! \ n')
            return
        else:
            self.nPassengers = nPassengers
            self.meanWeight  = meanWeight
    
    def  FillWithFuel (self , nLitres ) :
        self.fuel = nLitres
    
    def  AddPassengers (self ,  nPassengers ,  meanWeight = 65):
        totalPas = self.nPassengers + nPassengers
        if totalPas > self.capacity :
            print ( ' \n\tNo caben tantos pasajeros!!! \n ' )
            return
        else :
            self.nPassengers = totalPas
            self.meanWeight  = meanWeight
    
    def  GetTotalWeight(self):
        return self.weight + self.nPassengers*self.meanWeight + self.fuel*self.fuelDensity
    
    def  DoesItTakeOff(self):
        weight = self.GetTotalWeight()
        if weight < self.maxWeight :
            print ('\n\tDespega perfectamente!\n')
        else:
            print ('\n\tNo se levanta del suelo... \n')
    

    #El constructor de la clase:
    def __init__(self ,  name = '' ,  weight = 50e3 ,  capacity = 200 , maxWeight = 1e5, maxWeightAtLanding = 8e4, consumoalos100 = 2100, velocity =300):
        self.name = name
        self.weight = weight # OEW
        self.capacity = capacity
        self.maxWeight = maxWeight

        # Initial  parameters:
        self.fuel = 0 # In  liters, for example: 50 e3 liters
        self.nPassengers = 0
        self.meanWeight = 65 # kg
        self.fuelDensity = 0.8 # kg/L f o r  kerosene
        self.maxWeightAtLanding = maxWeightAtLanding
        self.consumoalos100 = consumoalos100
        self.velocity = velocity #km/h

#=======================================================================
#Probamos la clase, date cuenta que podriamos hacer un nuevo script e importar esta clase "from path/al/python import aeroplane"
#Vamos a definir un objeto de la clase aeroplane. Si hago un print del objeto me dice que pertenece a la clase Aeroplane

myPlane =  Aeroplane('A320') #con atributos por defecto
print(myPlane) 
myPlane_2 =  Aeroplane('A320',50e3,196,1.2e5)

print("Peso en vacio:")
myPlane_2.GetTotalWeight()

#ahora lo llenamos:
print("cargando el avion")
myPlane_2.FillWithFuel(50e3)
print('embarcando...')
myPlane_2.AddPassengers(200)
print('Volvemos a probar...')
myPlane_2.AddPassengers(160)

pesofinal = myPlane_2.GetTotalWeight()
print("Peso ahora:%.1f kg"%(pesofinal))
print("Despega?")
myPlane_2.DoesItTakeOff()

#Ejercicio
# a) Crea un método de la clase que calcule cuanto tiempo puede volar dada la cantidad de combustible   
# b) Crea otro método que nos diga si es seguro aterrizar dado el peso del avión (conociendo su maxWeightAtLanding)
# c) Obtén el tiempo de vuelo necesario para quemar el suficiente combustible y que no se super el maxWeightAtLanding


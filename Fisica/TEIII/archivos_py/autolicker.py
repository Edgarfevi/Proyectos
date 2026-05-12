import pyautogui
import keyboard
import time

# --- Configuración ---
tecla_activar = 's'       # Tecla para iniciar/pausar
tecla_salir = 'esc'       # Tecla para cerrar el programa
intervalo = 10.1    










































































































































      # Velocidad de clic
# ---------------------

print(f"Presiona '{tecla_activar}' para activar/pausar el autoclicker.")
print(f"Presiona '{tecla_salir}' para salir del programa.")

haciendo_clic = False

while True:
    # 1. Verificar si se quiere salir
    if keyboard.is_pressed(tecla_salir):
        print("Saliendo...")
        break

    # 2. Verificar si se presiona la tecla de alternar (toggle)
    if keyboard.is_pressed(tecla_activar):
        haciendo_clic = not haciendo_clic
        estado = "ACTIVADO" if haciendo_clic else "PAUSADO"
        print(f"Autoclicker {estado}")
        time.sleep(0.2) # Pequeña pausa para evitar rebote de tecla

    # 3. Ejecutar el clic si está activo
    if haciendo_clic:
        pyautogui.press('enter')
        time.sleep(intervalo)
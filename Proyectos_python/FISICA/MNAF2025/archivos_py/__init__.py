import time
import asyncio


async def tarea1():
    print("Tarea iniciada...")
    await asyncio.sleep(2)
    print("Tarea completada.")

def tarea2():
    print("Tarea 2 iniciada...")
    time.sleep(1)
    print("Tarea 2 completada.")

async def main():
    print("Starting main function...")
    await tarea1()
    tarea2()
    print("Main function completed.")
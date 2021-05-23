import numpy 
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class Harmonic:
    '''
    Источник, создающий гармонический сигнал
    '''
    def __init__(self, Nl, eps=1.0, mu=1.0, phi_0=None, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        Nl - количество отсчетов на длину волны;
        Sc - число Куранта.
        '''
        
        self.Nl = Nl
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

        if phi_0 is None:
            self.phi_0 = -2 * numpy.pi / Nl
        else:
            self.phi_0 = phi_0

    def getField(self, m, q):
        return self.magnitude * numpy.sin(2 * numpy.pi / self.Nl *
                (self.Sc * q - m * numpy.sqrt(self.mu * self.eps))+ self.phi_0)
    

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 2100

    # Размер области моделирования вдоль оси X в метрах
    X = 2.5
    #Размер ячейки разбиения
    dx = 5e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos + 50]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = 6.0

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = Harmonic(120.0, eps[sourcePos], mu[sourcePos])

     # Ez[1] в предыдущий момент времени
    oldEzLeft = Ez[1]

    # Расчет коэффициентов для граничных условий
    tempLeft = Sc / numpy.sqrt(mu[0] * eps[0])
    koeffABCLeft = (tempLeft - 1) / (tempLeft + 1)
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.0
    display_ymax = 2.0

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия ABC первой степени (слева)
        Ez[0] = oldEzLeft + koeffABCLeft * (Ez[1] - Ez[0])
        oldEzLeft = Ez[1]

        # Граничные условия PEC (справа)
        Ez[-1] = 0
        
        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 10 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -2.0, 2.0, dt)

    # Отображение спектра сигнала
    tools.Furie(2 ** 18, probe.E, dt).FFT()


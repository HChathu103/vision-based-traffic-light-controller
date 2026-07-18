import random
import time
import threading
import pygame
import sys

# Default values of signal timers
defaultGreen = {0: 10, 1: 10}  # 0: West, 1: South
defaultRed = 20
defaultYellow = 3

signals = []
noOfSignals = 2
currentGreen = 0   # 0: West (Right-moving), 1: South (Up-moving)
nextGreen = (currentGreen + 1) % noOfSignals
currentYellow = 0   

# Vehicle speeds
speeds = {'design': 1.2}  

# --- 700x700 Screen coordinates configured for perfect single-lane alignment ---
# Adjusted 'up' to 340 so vehicles line up perfectly on the right-hand side track
x = {
    'right': [-50, -50, -50],           # Spawning from West (moving Right)
    'up': [440, 440, 440],              # Spawning from South aligned to the right side lane
}    

y = {
    'right': [180, 180, 180],           # Westbound incoming lane (Centered at y=340)
    'up': [750, 750, 750],              # Southbound incoming lane
}

# Turn point set higher up (y=200) to force the vehicle to step forward after the stop line (y=370)
turnPointY = 200

# Tracking queues per direction (Single Lane: index 0 used)
vehicles = {
    'right': {0: [], 1: [], 2: [], 'crossed': 0},
    'up': {0: [], 1: [], 2: [], 'crossed': 0}
}

vehicleTypes = {0: 'design', 1: 'design', 2: 'design', 3: 'design'}
directionNumbers = {0: 'right', 1: 'up'}

# Stop lines 
stopLines = {'right': 290, 'up': 370}
defaultStop = {'right': 280, 'up': 380}

signalCoods = [(270, 25), (600, 350)]
signalTimerCoods = [(270, 5), (600, 330)]

stoppingGap = 50 
movingGap = 50    

pygame.init()
simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.signalText = ""
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.turned = False
        
        vehicles[direction][lane].append(self)
        self.index = len(vehicles[direction][lane]) - 1
        
        # Load default surface if image asset is missing
        try:
            path = "images/" + direction + "/" + vehicleClass + ".png"
            self.image = pygame.image.load(path)
        except:
            if direction == 'right':
                self.image = pygame.Surface((35, 20))  # Horizontal block
            else:
                self.image = pygame.Surface((20, 35))  # Vertical block
            self.image.fill((255, 230, 0)) # Yellow box representation

        # Collision avoidance queue logic
        if len(vehicles[direction][lane]) > 1:
            lead_vehicle = vehicles[direction][lane][self.index - 1]
            if direction == 'right':
                self.stop = lead_vehicle.stop - 55
            elif direction == 'up':
                self.stop = lead_vehicle.stop + 55
        else:
            self.stop = defaultStop[direction]
            
        # Offset next spawning coordinate to queue up off-screen
        if direction == 'right':
            x[direction][lane] -= 65
        elif direction == 'up':
            y[direction][lane] += 65
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.image, (self.x, self.y))

    def move(self):
        # Once turned or cleared, all vehicles move straight towards East (right)
        if self.turned:
            self.x += self.speed
            return

        lead_vehicle = vehicles[self.direction][self.lane][self.index - 1] if self.index > 0 else None

        if self.direction == 'right':
            if self.x >= 350:
                self.turned = True
                return

            if self.crossed == 0 and self.x + 35 > stopLines[self.direction]:
                self.crossed = 1

            allow_move = (
                self.x + 35 <= self.stop or 
                self.crossed == 1 or 
                (currentGreen == 0 and currentYellow == 0)
            )
            no_collision = (lead_vehicle is None or lead_vehicle.turned or self.x + 35 < (lead_vehicle.x - movingGap))

            if allow_move and no_collision:
                self.x += self.speed

        elif self.direction == 'up':
            # South vehicles move past stopline (370) up to turnPointY (200) before turning right
            if self.y <= turnPointY:
                self.turned = True
                self.image = pygame.transform.rotate(self.image, -90)  # Rotate to face East
                self.y = 200  # Snap to horizontal center lane
                return

            if self.crossed == 0 and self.y < stopLines[self.direction]:
                self.crossed = 1

            allow_move = (
                self.y >= self.stop or 
                self.crossed == 1 or 
                (currentGreen == 1 and currentYellow == 0)
            )
            no_collision = (lead_vehicle is None or lead_vehicle.turned or self.y > (lead_vehicle.y + 35 + movingGap))

            if allow_move and no_collision:
                self.y -= self.speed

# Signal control loop logic
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen[0])
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red + ts1.yellow + ts1.green, defaultYellow, defaultGreen[1])
    signals.append(ts2)
    repeat()

def repeat():
    global currentGreen, currentYellow, nextGreen
    while signals[currentGreen].green > 0:   
        updateValues()
        time.sleep(1)
    currentYellow = 1   
    
    # Reset stops when signals cycle
    for direction in ['right', 'up']:
        for lane in range(1):
            for vehicle in vehicles[direction][lane]:
                vehicle.stop = defaultStop[direction]
        
    while signals[currentGreen].yellow > 0:  
        updateValues()
        time.sleep(1)
    currentYellow = 0   
    
    signals[currentGreen].green = defaultGreen[currentGreen]
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen 
    nextGreen = (currentGreen + 1) % noOfSignals    
    signals[nextGreen].red = signals[currentGreen].yellow + signals[currentGreen].green    
    repeat()  

def updateValues():
    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                signals[i].green -= 1
            else:
                signals[i].yellow -= 1
        else:
            signals[i].red -= 1

# Vehicle Generator from West (0) and South (1)
def generateVehicles():
    while True:
        lane_number = 0  
        direction_number = random.choice([0, 1])  # 0: West, 1: South
        Vehicle(lane_number, 'design', direction_number, directionNumbers[direction_number])
        time.sleep(3.0)

class Main:
    thread1 = threading.Thread(name="initialization", target=initialize, args=())    
    thread1.daemon = True
    thread1.start()

    black = (0, 0, 0)
    white = (255, 255, 255)

    screenWidth = 700
    screenHeight = 700
    screenSize = (screenWidth, screenHeight)

    screen = pygame.display.set_mode(screenSize)
    pygame.display.set_caption("T-JUNCTION ALIGNED SIMULATION")

    # FIX (supports bug 2 - "turn right only after moving up a little from
    # where it stopped"): there was no clock/FPS cap anywhere in the main
    # loop, so it ran as fast as the CPU allowed. The ~90px gap between the
    # stop point and turnPointY was always there in the code, but with an
    # uncapped loop it was covered in a tiny fraction of a real second, so it
    # visually looked like vehicles turned right immediately at the point
    # they had stopped. Capping the frame rate makes that "creep forward a
    # little, then turn right" motion actually visible, at a controlled,
    # consistent speed.
    clock = pygame.time.Clock()
    FPS = 60

    try:
        background = pygame.image.load('images/intersection1.png')
        use_bg_img = True
    except:
        use_bg_img = False

    try:
        redSignal = pygame.image.load('images/signals/red.png')
        yellowSignal = pygame.image.load('images/signals/yellow.png')
        greenSignal = pygame.image.load('images/signals/green.png')
    except:
        redSignal = pygame.Surface((20, 20)); redSignal.fill((255, 0, 0))
        yellowSignal = pygame.Surface((20, 20)); yellowSignal.fill((255, 255, 0))
        greenSignal = pygame.Surface((20, 20)); greenSignal.fill((0, 255, 0))

    font = pygame.font.Font(None, 24) 

    thread2 = threading.Thread(name="generateVehicles", target=generateVehicles, args=())    
    thread2.daemon = True
    thread2.start()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        if use_bg_img:
            screen.blit(background, (0, 0))
        else:
            # Render custom T-junction
            screen.fill((100, 150, 100))  # Green Grass
            
            # Roads
            pygame.draw.rect(screen, (70, 70, 70), (0, 310, 700, 80))    # West-East road
            pygame.draw.rect(screen, (70, 70, 70), (310, 390, 80, 310))  # South road
            
            # Lane dividers
            pygame.draw.line(screen, (150, 150, 150), (0, 350), (700, 350), 2)
            pygame.draw.line(screen, (150, 150, 150), (350, 390), (350, 700), 2)
            
            # White Stop lines
            pygame.draw.line(screen, (255, 255, 255), (290, 310), (290, 390), 3) # West Stop Line
            pygame.draw.line(screen, (255, 255, 255), (310, 410), (390, 410), 3) # South Stop Line (y=410)

        # Rendering Signals
        for i in range(noOfSignals):  
            if i == currentGreen:
                if currentYellow == 1:
                    signals[i].signalText = signals[i].yellow
                    screen.blit(yellowSignal, signalCoods[i])
                else:
                    signals[i].signalText = signals[i].green
                    screen.blit(greenSignal, signalCoods[i])
            else:
                if signals[i].red <= 10:
                    signals[i].signalText = signals[i].red
                else:
                    signals[i].signalText = "---"
                screen.blit(redSignal, signalCoods[i])

        # Rendering Signal Timer Texts
        signalTexts = ["", ""]
        for i in range(noOfSignals):  
            signalTexts[i] = font.render(str(signals[i].signalText), True, white, black)
            screen.blit(signalTexts[i], signalTimerCoods[i])

        # Movement loop and rendering of active vehicles
        for vehicle in list(simulation):  
            vehicle.move()
            if vehicle.x > 720:  
                simulation.remove(vehicle)
                vehicles[vehicle.direction][vehicle.lane].remove(vehicle)
                for idx, veh in enumerate(vehicles[vehicle.direction][vehicle.lane]):
                    veh.index = idx
            else:
                screen.blit(vehicle.image, [vehicle.x, vehicle.y])
                
        pygame.display.update()
        clock.tick(FPS)

Main()
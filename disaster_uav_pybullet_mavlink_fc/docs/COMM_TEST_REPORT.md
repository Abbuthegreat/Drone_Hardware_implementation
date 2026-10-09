# Communication test: PASS

| # | step | result | evidence |
|---|---|---|---|
| 1 | PyBullet + simulated FC MAVLink server started | **PASS** | pid 78136, udpin:0.0.0.0:14540, world=earthquake scene |
| 2 | ROS 2 <-> MAVLink bridge node started | **PASS** | pid 75228, connection udp:127.0.0.1:14540 |
| 3 | HEARTBEAT received from FC | **PASS** | bridge log: 'FC heartbeat received (sys 1)'; /uav/mode publishing |
| 4 | telemetry received (/uav/telemetry, /uav/odom, /uav/imu, /uav/gps/fix) | **PASS** | telemetry 30.5 Hz, odom n=91, imu n=315, gps n=30, mode=HOLD, armed=False |
| 5 | arm command sent (ROS service /uav/arm -> MAVLink COMMAND_LONG) | **PASS** | service: success=True message=MAV_RESULT=0; telemetry armed=True |
| 6 | COMMAND_ACK verified | **PASS** | ROS response MAV_RESULT=0 (ACCEPTED); FC log: 'TX COMMAND_ACK command=400 result=0' |
| 7 | setpoints streamed continuously by the bridge (offboard-style) | **PASS** | 20.4 Hz measured from bridge counter (configured 20 Hz) |
| 8 | OFFBOARD mode accepted | **PASS** | service success=True, telemetry mode=OFFBOARD |
| 9 | vehicle responds physically to /uav/cmd_vel (climb) | **PASS** | PyBullet altitude +4.74 m, vD=-1.02 m/s, total motor thrust 21.9 N; ROS telemetry z=4.86 |
| 10 | velocity command changed -> acceleration + displacement observed | **PASS** | ENU +y (North) cmd 1.5 m/s: displacement 5.49 m, initial accel 1.28 m/s^2, final vN=1.52 m/s, pitch -0.9 deg (nose-down) |
| 11 | ros2 CLI: topic list / echo / pub cause real motion | **PASS** | `ros2 topic list` shows /uav/cmd_vel + /uav/telemetry=True; `topic echo --once /uav/telemetry` returned JSON=True; `ros2 topic pub /uav/cmd_vel linear.x=1.5 (East)` -> PyBullet East displacement 4.57 m |
| 12 | stop commands (setpoint stream stopped) | **PASS** | speed at stop 1.50 m/s; setpoint streaming OFF |
| 13 | failsafe / hold when setpoints stop | **PASS** | mode FAILSAFE_HOLD after 0.77 s (FC timeout 0.5 s + telemetry latency); speed 1.50->0.01 m/s; still airborne z=5.16 m; FC log: 'Failsafe: OFFBOARD_SETPOINT_LOSS' |
| 14 | land command -> vehicle descends and lands | **PASS** | LAND accepted=True; landed flag=True, altitude 0.15 m |
| 15 | disarm | **PASS** | service success=True MAV_RESULT=0 |
| 16 | motor thrust falls after disarm | **PASS** | armed=False, total thrust 0.000 N (hover ~21.6 N), max rpm 0; ROS telemetry armed=False |

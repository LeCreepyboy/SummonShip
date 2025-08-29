using OWML.Common;
using OWML.ModHelper;
using OWML.Utils;
using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

namespace SummonShip
{
    public class SummonShip : ModBehaviour
    {
        // --- Global parameters ---
        const float UPWARD_BURN_SECONDS = 7f;    // planetary ascent duration (was 10f)
        const float LEAD_TIME = 0.00f;           // target = player position + v*lead (0 = none)

        // Auto cutoff near player
        const float AUTO_CUTOFF_DISTANCE = 13f;

        // Launch safety: block start if player is too close to ship
        const float SAFETY_MIN_DISTANCE = 30f;

        // Hold-to-trigger threshold for X key
        const float HOLD_THRESHOLD_SECONDS = 0.5f;

        // Backstop (unused directly but kept for clarity)
        const float STOP_DISTANCE = 1.0f;
        const float STOP_VEL_EPS = 0.15f;

        // --- Desired speed shaping (fast far, gentle near) ---
        const float VR_SLOPE_FAR = 1.00f;
        const float VR_MIN_FAR = 30f;
        const float VR_MAX = 300f;

        const float VR_NEAR_DIST = 20f;
        const float VR_NEAR_MAX = 14f;

        const float VT_MAX = 30f;   // lateral speed cap
        const float TAU_TANG = 1.2f;  // seconds to damp lateral error

        // Adaptive braking (v^2 / 2a)
        const float BRAKE_BUFFER = 12f;
        const float BRAKE_MARGIN = 0.85f;
        const float A_R_FAR = 45f;   // radial accel far
        const float A_R_NEAR = 18f;  // radial accel near
        const float A_T_FAR = 14f;   // tangential accel far
        const float A_T_NEAR = 8f;   // tangential accel near
        const float BRAKE_BLEND_DIST = 200f;  // distance where near gains fade in

        // Reduce tangential authority at close range
        const float TANG_WEIGHT_MAX_DIST = 80f;

        // Velocity error -> thruster input
        const float K_VEL_CMD_FAR = 0.11f;
        const float K_VEL_CMD_NEAR = 0.17f;
        const float KCMD_NEAR_DIST = 60f;

        // Desired-velocity smoothing
        const float VDES_SMOOTH_FAR = 0.25f;
        const float VDES_SMOOTH_NEAR = 0.55f;
        const float VDES_SMOOTH_DIST = 120f;

        // Command input smoothing & clamping
        const float MAX_INPUT_MAG = 1.0f;
        const float CLOSE_DIST_SOFTEN = 1.5f;
        const float CLOSE_MAX_INPUT = 0.58f;
        const float CONTROL_SMOOTH_ALPHA = 0.30f;
        const float MAX_DELTA_PER_STEP = 0.28f;

        // Small deadzone near target to reduce micro-oscillation
        const float VERR_DEADZONE_DIST = 25f;
        const float VERR_DEADZONE_MAG = 0.20f;

        // --- Attitude control (airlock down toward player) ---
        const float ORIENT_KP = 6.0f;  // proportional gain on angle
        const float ORIENT_KD = 1.8f;  // damping on angular velocity
        const float ORIENT_MAX_INPUT = 1.0f;  // clamp rotational input
        const float ORIENT_SMOOTH_ALPHA = 0.35f; // rotation command smoothing
        const float ORIENT_MAX_DELTA = 0.50f; // jerk cap for rotation
        const float ORIENT_FADE_NEAR_MIN = 8f;    // fade attitude torque below this
        const float ORIENT_FADE_NEAR_MAX = 58f;   // to this distance

        // --- Valid bodies for planetary ascent ---
        static readonly AstroObject.Name[] VALID_PLANETS = new[]
        {
            AstroObject.Name.TimberHearth,
            AstroObject.Name.BrittleHollow,
            AstroObject.Name.CaveTwin,   // Ember Twin
            AstroObject.Name.TowerTwin,  // Ash Twin
            AstroObject.Name.GiantsDeep,
            AstroObject.Name.DarkBramble
        };

        // --- State ---
        bool counterActive = false;   // true during planetary ascent
        bool isTimerDone = false;

        Coroutine approachRoutine;
        Vector3 _uPrevLocal = Vector3.zero; // smoothed translational command (local)
        Vector3 _vDesPrevWS = Vector3.zero; // smoothed desired velocity (world)
        Vector3 _wPrevLocal = Vector3.zero; // smoothed rotational command (local)

        // Hold-X state
        bool _xHeld = false;
        float _xHeldTime = 0f;
        bool _xHoldTriggered = false; // fires once per hold

        // --- Utils ---
        static float Lerp01(float a, float b, float t) => a + (b - a) * Mathf.Clamp01(t);

        static float DistBlend(float dist, float farVal, float nearVal, float blendDist)
        {
            // Returns farVal far away, blends toward nearVal as distance decreases.
            float wNear = 1f - Mathf.Clamp01(dist / Mathf.Max(1f, blendDist));
            return Lerp01(farVal, nearVal, wNear);
        }

        static float SpeedLimitForStoppingAdaptive(float dist, float a, float buffer, float margin)
        {
            // Speed such that stopping distance (plus buffer) fits in 'dist'
            float usable = Mathf.Max(0f, dist - buffer);
            float v = Mathf.Sqrt(2f * Mathf.Max(0.01f, a) * usable);
            return margin * v;
        }

        static Vector3 ProjectOnPlane(Vector3 v, Vector3 normal)
        {
            return v - Vector3.Dot(v, normal) * normal;
        }

        // Check if ship is inside a valid planet GravityVolume (sphere trigger).
        // If yes, outputs planetUp = (planet center -> ship).worldNormalized
        static bool TryGetPlanetUpIfInsideValidPlanetGravity(Vector3 shipPos, out Vector3 planetUp)
        {
            planetUp = Vector3.up;

            foreach (var name in VALID_PLANETS)
            {
                var ao = Locator.GetAstroObject(name);
                if (ao == null) continue;

                var gvs = ao.GetComponentsInChildren<GravityVolume>(true);
                if (gvs == null || gvs.Length == 0) continue;

                foreach (var gv in gvs)
                {
                    if (gv == null) continue;

                    var spheres = gv.GetComponentsInChildren<SphereCollider>(true);
                    if (spheres == null || spheres.Length == 0) continue;

                    foreach (var sc in spheres)
                    {
                        if (sc == null || !sc.enabled) continue;

                        Vector3 worldCenter = sc.transform.TransformPoint(sc.center);
                        float scale = Mathf.Max(sc.transform.lossyScale.x, Mathf.Max(sc.transform.lossyScale.y, sc.transform.lossyScale.z));
                        float worldRadius = sc.radius * scale;

                        float d2 = (shipPos - worldCenter).sqrMagnitude;
                        if (d2 <= worldRadius * worldRadius)
                        {
                            Vector3 centerApprox = ao.transform.position;
                            Vector3 dir = shipPos - centerApprox;
                            if (dir.sqrMagnitude < 1e-6f) dir = shipPos - worldCenter;
                            planetUp = dir.normalized;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        // --- Coroutines ---
        IEnumerator ThrusterTiming()
        {
            yield return new WaitForSeconds(UPWARD_BURN_SECONDS);
            isTimerDone = true;
        }

        IEnumerator ApproachVelocityTracking()
        {
            var shipBody = Locator.GetShipBody();
            var shipRB = shipBody; // OWRigidbody
            var thruster = shipBody.GetComponent<ThrusterModel>();

            shipBody.EnableCollisionDetection();

            var notifStart = new NotificationData(NotificationTarget.Player,
                "Summon: approach mode", 5f, true);
            NotificationManager.SharedInstance.PostNotification(notifStart, false);

            _uPrevLocal = Vector3.zero;
            _wPrevLocal = Vector3.zero;
            _vDesPrevWS = shipRB.GetVelocity();

            while (true)
            {
                var playerBody = Locator.GetPlayerBody();
                var playerT = Locator.GetPlayerTransform();
                if (playerBody == null || playerT == null) yield break;

                // Player state
                Vector3 pPos = playerT.position;
                Vector3 pVel = playerBody.GetVelocity();

                // Target = player position (+ lead if any)
                Vector3 targetPos = pPos + pVel * LEAD_TIME;
                Vector3 targetVel = pVel;

                // Ship state
                Vector3 sPos = shipBody.transform.position;
                Vector3 sVel = shipRB.GetVelocity();

                // Geometry
                Vector3 rWS = targetPos - sPos;
                float dist = rWS.magnitude;
                if (dist < 1e-4f) dist = 1e-4f;
                Vector3 dirR = rWS / dist;

                // Final message at end distance
                if (dist <= AUTO_CUTOFF_DISTANCE)
                {
                    CancelAutopilot("Summon: Player abduction protocol, please don't move");
                    yield break;
                }

                // Radial/tangential split
                Vector3 radialWS = dirR * dist;
                Vector3 tangErrWS = rWS - radialWS;
                Vector3 dirT = tangErrWS.sqrMagnitude > 1e-6f ? tangErrWS.normalized : Vector3.zero;

                // Desired speeds before stop limits
                float v_r_far = (dist > VR_NEAR_DIST)
                    ? Mathf.Clamp(VR_SLOPE_FAR * dist, VR_MIN_FAR, VR_MAX)
                    : Mathf.Lerp(0f, VR_NEAR_MAX, dist / VR_NEAR_DIST);

                float v_t_far = (dirT == Vector3.zero) ? 0f
                                 : Mathf.Min(tangErrWS.magnitude / Mathf.Max(0.2f, TAU_TANG), VT_MAX);

                // Stop-aware limits (adaptive braking)
                float a_r = DistBlend(dist, A_R_FAR, A_R_NEAR, BRAKE_BLEND_DIST);
                float a_t = DistBlend(dist, A_T_FAR, A_T_NEAR, BRAKE_BLEND_DIST);

                float v_r_allowed = SpeedLimitForStoppingAdaptive(dist - STOP_DISTANCE, a_r, BRAKE_BUFFER, BRAKE_MARGIN);
                float v_t_allowed = SpeedLimitForStoppingAdaptive(dist, a_t, BRAKE_BUFFER * 0.5f, BRAKE_MARGIN);

                float v_r_cmd = Mathf.Min(v_r_far, v_r_allowed);
                float v_t_cmd = Mathf.Min(v_t_far, v_t_allowed);

                // Fade tangential authority near target (quadratic)
                float tWeightLin = Mathf.Clamp01(dist / TANG_WEIGHT_MAX_DIST);
                float tangWeight = tWeightLin * tWeightLin;
                v_t_cmd *= tangWeight;

                // Desired velocity in world (raw)
                Vector3 v_des_raw = targetVel
                                  + v_r_cmd * dirR
                                  + (dirT != Vector3.zero ? v_t_cmd * (-dirT) : Vector3.zero);

                // Smooth desired velocity
                float vdesAlpha = Lerp01(VDES_SMOOTH_FAR, VDES_SMOOTH_NEAR, 1f - Mathf.Clamp01(dist / VDES_SMOOTH_DIST));
                Vector3 v_des = Vector3.Lerp(_vDesPrevWS, v_des_raw, vdesAlpha);
                _vDesPrevWS = v_des;

                // Velocity error -> thrusters
                float kcmd = Lerp01(K_VEL_CMD_FAR, K_VEL_CMD_NEAR, 1f - Mathf.Clamp01(dist / KCMD_NEAR_DIST));
                Vector3 v_err = v_des - sVel;

                if (dist <= VERR_DEADZONE_DIST && v_err.magnitude < VERR_DEADZONE_MAG)
                    v_err = Vector3.zero;

                Vector3 uWS = kcmd * v_err;

                // World -> local, smooth, clamp, apply
                Vector3 uLocalDesired = shipBody.transform.InverseTransformDirection(uWS);
                Vector3 uLocal = Vector3.Lerp(_uPrevLocal, uLocalDesired, CONTROL_SMOOTH_ALPHA);
                Vector3 du = uLocal - _uPrevLocal;
                if (du.magnitude > MAX_DELTA_PER_STEP)
                    uLocal = _uPrevLocal + du.normalized * MAX_DELTA_PER_STEP;

                float maxMag = (dist < CLOSE_DIST_SOFTEN) ? CLOSE_MAX_INPUT : MAX_INPUT_MAG;
                if (uLocal.magnitude > maxMag) uLocal = uLocal.normalized * maxMag;

                thruster.AddTranslationalInput(uLocal);
                _uPrevLocal = uLocal;

                // Attitude: align -ship.up with dirR (airlock toward player)
                Vector3 targetUp = -dirR;

                Vector3 fwdCandidate = ProjectOnPlane(targetVel, targetUp);
                if (fwdCandidate.sqrMagnitude < 1e-4f)
                    fwdCandidate = ProjectOnPlane(shipBody.transform.forward, targetUp);
                Vector3 targetFwd = fwdCandidate.sqrMagnitude > 1e-6f ? fwdCandidate.normalized : shipBody.transform.forward;

                Quaternion targetRot = Quaternion.LookRotation(targetFwd, targetUp);
                Quaternion curRot = shipBody.transform.rotation;

                Quaternion qErr = targetRot * Quaternion.Inverse(curRot);
                qErr.ToAngleAxis(out float angleDeg, out Vector3 axisWS);
                if (angleDeg > 180f) { angleDeg -= 360f; }
                float angleRad = angleDeg * Mathf.Deg2Rad;

                Vector3 wErrWS = axisWS.normalized * angleRad;
                Vector3 wCurWS = shipRB.GetAngularVelocity();

                float orientWeight = Mathf.Clamp01((dist - ORIENT_FADE_NEAR_MIN) / (ORIENT_FADE_NEAR_MAX - ORIENT_FADE_NEAR_MIN));

                Vector3 torqueWS = ORIENT_KP * wErrWS - ORIENT_KD * wCurWS;
                torqueWS *= orientWeight;

                Vector3 rotLocalDesired = shipBody.transform.InverseTransformDirection(torqueWS);

                Vector3 rotLocal = Vector3.Lerp(_wPrevLocal, rotLocalDesired, ORIENT_SMOOTH_ALPHA);
                Vector3 dw = rotLocal - _wPrevLocal;
                if (dw.magnitude > ORIENT_MAX_DELTA)
                    rotLocal = _wPrevLocal + dw.normalized * ORIENT_MAX_DELTA;

                if (rotLocal.magnitude > ORIENT_MAX_INPUT)
                    rotLocal = rotLocal.normalized * ORIENT_MAX_INPUT;

                thruster.AddRotationalInput(rotLocal);
                _wPrevLocal = rotLocal;

                yield return new WaitForFixedUpdate();
            }
        }

        // Cancel everything and re-enable collisions
        void CancelAutopilot(string reason = "Summon: aborted")
        {
            if (approachRoutine != null)
            {
                StopCoroutine(approachRoutine);
                approachRoutine = null;
            }
            counterActive = false;
            isTimerDone = false;
            _uPrevLocal = Vector3.zero;
            _wPrevLocal = Vector3.zero;
            _vDesPrevWS = Vector3.zero;
            Locator.GetShipBody().EnableCollisionDetection();

            var notif = new NotificationData(NotificationTarget.Player, reason, 5f, true);
            NotificationManager.SharedInstance.PostNotification(notif, false);
        }

        bool IsAutopilotActive() => counterActive || approachRoutine != null;

        // True if player is piloting in cockpit
        bool IsPlayerAtShipControls()
        {
            return OWInput.GetInputMode() == InputMode.ShipCockpit;
        }

        // Attempt to start Summon sequence
        void TryStartAutopilot()
        {
            // If player is already piloting, do nothing (no user message)
            if (IsPlayerAtShipControls()) return;

            // Launch safety: player too close to ship
            var shipBody = Locator.GetShipBody();
            var playerT = Locator.GetPlayerTransform();
            if (shipBody != null && playerT != null)
            {
                float dist = Vector3.Distance(playerT.position, shipBody.transform.position);
                if (dist <= SAFETY_MIN_DISTANCE)
                {
                    var warn = new NotificationData(NotificationTarget.Player,
                        "Summon ERROR: player too close", 5f, true);
                    NotificationManager.SharedInstance.PostNotification(warn, false);
                    return;
                }
            }

            // Decide planetary ascent only if inside a valid planet GravityVolume
            bool doPlanetaryClimb = false;
            Vector3 dummyUp;
            var sb = Locator.GetShipBody();
            if (sb != null)
            {
                doPlanetaryClimb = TryGetPlanetUpIfInsideValidPlanetGravity(sb.transform.position, out dummyUp);
            }

            if (doPlanetaryClimb)
            {
                var notif = new NotificationData(NotificationTarget.Player,
                    "Summon: Ship on a planet, beginning liftoff", 5f, true);
                NotificationManager.SharedInstance.PostNotification(notif, false);

                Locator.GetShipBody().DisableCollisionDetection();

                counterActive = true;
                isTimerDone = false;
                StartCoroutine(ThrusterTiming());
            }
            else
            {
                var notif = new NotificationData(NotificationTarget.Player,
                    "Summon: Ship in the wilds, direct approach", 5f, true);
                NotificationManager.SharedInstance.PostNotification(notif, false);

                approachRoutine = StartCoroutine(ApproachVelocityTracking());
            }
        }

        // --- Hooks ---
        private void Start()
        {
            LoadManager.OnCompleteSceneLoad += (scene, loadScene) =>
            {
                if (loadScene != OWScene.SolarSystem) return;
                ModHelper.Console.WriteLine("[SummonShip] Hold X ≥0.5s to toggle. vmax=300, cutoff=13 m, safety<30 m.", MessageType.Info);
            };
        }

        // --- Main loop ---
        private void Update()
        {
            if (Keyboard.current == null) return;

            // Hold-to-toggle on X
            var xKey = Keyboard.current.xKey;
            if (xKey.wasPressedThisFrame)
            {
                _xHeld = true;
                _xHeldTime = 0f;
                _xHoldTriggered = false;
            }

            if (_xHeld && xKey.isPressed)
            {
                _xHeldTime += Time.deltaTime;

                if (!_xHoldTriggered && _xHeldTime >= HOLD_THRESHOLD_SECONDS)
                {
                    _xHoldTriggered = true;

                    if (IsAutopilotActive())
                    {
                        CancelAutopilot("Summon: aborted");
                    }
                    else
                    {
                        TryStartAutopilot();
                    }
                }
            }

            if (xKey.wasReleasedThisFrame)
            {
                _xHeld = false;
                _xHeldTime = 0f;
                _xHoldTriggered = false;
            }

            // Planetary ascent phase: thrust along local(direction of planetUp world)
            if (counterActive)
            {
                var shipBody = Locator.GetShipBody();
                var thruster = shipBody.GetComponent<ThrusterModel>();

                Vector3 upDirWS;
                bool hasPlanetUp = TryGetPlanetUpIfInsideValidPlanetGravity(shipBody.transform.position, out upDirWS);
                if (!hasPlanetUp)
                {
                    // Lost planetary volume during ascent -> switch to approach
                    counterActive = false;
                    isTimerDone = false;

                    var notif = new NotificationData(NotificationTarget.Player,
                        "Summon: planetary volume lost; approach mode", 5f, true);
                    NotificationManager.SharedInstance.PostNotification(notif, false);

                    approachRoutine = StartCoroutine(ApproachVelocityTracking());
                }
                else
                {
                    // World -> local for thruster model
                    Vector3 upDirLocal = shipBody.transform.InverseTransformDirection(upDirWS).normalized;
                    thruster.AddTranslationalInput(upDirLocal);
                }
            }

            // Transition to approach after ascent timer (no extra message to avoid duplication)
            if (isTimerDone)
            {
                counterActive = false;
                isTimerDone = false;
                approachRoutine = StartCoroutine(ApproachVelocityTracking());
            }
        }
    }
}

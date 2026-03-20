/// Central configuration for the Netra AI capture app.
class AppConfig {
  AppConfig._();

  // ── API ──────────────────────────────────────────────────────────────
  /// Base URL for the Netra AI backend. Change this per environment.
  static const String apiBaseUrl = 'https://api.netra-ai.com/v1';

  /// Timeout durations for HTTP calls (milliseconds).
  static const int connectTimeout = 15000;
  static const int receiveTimeout = 30000;
  static const int uploadTimeout = 120000;

  // ── App Metadata ─────────────────────────────────────────────────────
  static const String appName = 'Netra AI';
  static const String appVersion = '1.0.0';
  static const String appTagline = 'AI-Powered Diabetic Retinopathy Screening';

  // ── Image Capture ────────────────────────────────────────────────────
  /// Maximum image file size in bytes (10 MB).
  static const int maxImageSizeBytes = 10 * 1024 * 1024;

  /// Accepted image MIME types.
  static const List<String> acceptedImageTypes = [
    'image/jpeg',
    'image/png',
  ];

  /// Target image quality for JPEG compression (0-100).
  static const int imageQuality = 90;

  // ── Offline Queue ────────────────────────────────────────────────────
  /// Max items kept in the Hive offline queue before oldest are dropped.
  static const int maxOfflineQueueSize = 200;

  /// Interval between automatic sync attempts (seconds).
  static const int syncIntervalSeconds = 60;

  // ── Screening ────────────────────────────────────────────────────────
  /// DR severity labels mapped to integer grades.
  static const Map<int, String> drGradeLabels = {
    0: 'No DR',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'Proliferative DR',
  };

  /// Risk-level color keys for the traffic-light display.
  static const Map<String, String> riskLevels = {
    'low': 'No immediate concern',
    'moderate': 'Follow-up recommended',
    'high': 'Urgent referral needed',
  };
}

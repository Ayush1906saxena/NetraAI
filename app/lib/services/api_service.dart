import 'package:dio/dio.dart';

import '../config/api_client.dart';
import '../models/patient.dart';
import '../models/screening.dart';
import '../models/report.dart';

/// Handles all HTTP calls to the Netra AI backend.
class ApiService {
  final Dio _dio = ApiClient().dio;

  // ── Screening CRUD ───────────────────────────────────────────────────

  /// Create a new screening record on the server.
  Future<Screening> createScreening(Patient patient) async {
    final response = await _dio.post(
      '/screenings',
      data: {'patient': patient.toJson()},
    );
    return Screening.fromJson(response.data as Map<String, dynamic>);
  }

  /// Upload a fundus image for the given eye.
  ///
  /// [onProgress] receives a 0.0 – 1.0 fraction.
  Future<EyeImage> uploadImage(
    String screeningId,
    Eye eye,
    String filePath, {
    void Function(double)? onProgress,
  }) async {
    final formData = FormData.fromMap({
      'eye': eye == Eye.left ? 'left' : 'right',
      'image': await MultipartFile.fromFile(
        filePath,
        filename: '${eye == Eye.left ? 'left' : 'right'}_fundus.jpg',
      ),
    });

    final response = await _dio.post(
      '/screenings/$screeningId/images',
      data: formData,
      options: Options(contentType: 'multipart/form-data'),
      onSendProgress: (sent, total) {
        if (total > 0 && onProgress != null) {
          onProgress(sent / total);
        }
      },
    );

    return EyeImage.fromJson(response.data as Map<String, dynamic>);
  }

  // ── Analysis ─────────────────────────────────────────────────────────

  /// Tell the server to start AI analysis for the screening.
  Future<void> triggerAnalysis(String screeningId) async {
    await _dio.post('/screenings/$screeningId/analyze');
  }

  /// Fetch analysis results. Returns `null` if not ready yet.
  Future<Report?> getResults(String screeningId) async {
    try {
      final response = await _dio.get(
        '/screenings/$screeningId/results',
      );
      if (response.statusCode == 200 && response.data != null) {
        return Report.fromJson(response.data as Map<String, dynamic>);
      }
      return null;
    } on DioException catch (e) {
      // 404 means results are not ready yet.
      if (e.response?.statusCode == 404) return null;
      rethrow;
    }
  }

  // ── Report ───────────────────────────────────────────────────────────

  /// Request PDF report generation. Returns the download URL.
  Future<String> generateReport(String screeningId) async {
    final response = await _dio.post(
      '/screenings/$screeningId/report',
    );
    return response.data['report_url'] as String;
  }

  // ── History ──────────────────────────────────────────────────────────

  /// Get the most recent screenings for the logged-in operator.
  Future<List<Screening>> getRecentScreenings({int limit = 20}) async {
    final response = await _dio.get(
      '/screenings',
      queryParameters: {'limit': limit, 'sort': '-created_at'},
    );
    final list = response.data as List<dynamic>;
    return list
        .map((e) => Screening.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  /// Search screenings by patient name or phone.
  Future<List<Screening>> searchScreenings(String query) async {
    final response = await _dio.get(
      '/screenings/search',
      queryParameters: {'q': query},
    );
    final list = response.data as List<dynamic>;
    return list
        .map((e) => Screening.fromJson(e as Map<String, dynamic>))
        .toList();
  }
}

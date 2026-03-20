import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import 'app_config.dart';

/// Singleton Dio HTTP client with JWT interceptor and standard config.
class ApiClient {
  ApiClient._internal();
  static final ApiClient _instance = ApiClient._internal();
  factory ApiClient() => _instance;

  final FlutterSecureStorage _secureStorage = const FlutterSecureStorage();

  late final Dio dio = _createDio();

  Dio _createDio() {
    final dio = Dio(
      BaseOptions(
        baseUrl: AppConfig.apiBaseUrl,
        connectTimeout: const Duration(milliseconds: AppConfig.connectTimeout),
        receiveTimeout: const Duration(milliseconds: AppConfig.receiveTimeout),
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'X-App-Version': AppConfig.appVersion,
        },
      ),
    );

    // JWT interceptor — attaches Bearer token to every request.
    dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final token = await _secureStorage.read(key: 'jwt_token');
          if (token != null && token.isNotEmpty) {
            options.headers['Authorization'] = 'Bearer $token';
          }
          return handler.next(options);
        },
        onError: (error, handler) async {
          // On 401, clear stored token so the app can re-authenticate.
          if (error.response?.statusCode == 401) {
            await _secureStorage.delete(key: 'jwt_token');
          }
          return handler.next(error);
        },
      ),
    );

    // Logging interceptor (debug builds only).
    assert(() {
      dio.interceptors.add(LogInterceptor(
        requestBody: true,
        responseBody: true,
        logPrint: (obj) => print('[API] $obj'),
      ));
      return true;
    }());

    return dio;
  }

  /// Store a new JWT token after login / refresh.
  Future<void> setToken(String token) async {
    await _secureStorage.write(key: 'jwt_token', value: token);
  }

  /// Remove the stored JWT token on logout.
  Future<void> clearToken() async {
    await _secureStorage.delete(key: 'jwt_token');
  }

  /// Read the current stored token (may be null).
  Future<String?> getToken() async {
    return _secureStorage.read(key: 'jwt_token');
  }
}

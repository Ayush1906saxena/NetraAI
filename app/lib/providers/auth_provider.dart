import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../config/api_client.dart';

/// Authentication state.
class AuthState {
  final bool isAuthenticated;
  final String? token;
  final String? operatorName;
  final bool isLoading;
  final String? error;

  const AuthState({
    this.isAuthenticated = false,
    this.token,
    this.operatorName,
    this.isLoading = false,
    this.error,
  });

  AuthState copyWith({
    bool? isAuthenticated,
    String? token,
    String? operatorName,
    bool? isLoading,
    String? error,
  }) {
    return AuthState(
      isAuthenticated: isAuthenticated ?? this.isAuthenticated,
      token: token ?? this.token,
      operatorName: operatorName ?? this.operatorName,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }
}

/// Manages JWT-based authentication for the operator.
class AuthNotifier extends StateNotifier<AuthState> {
  AuthNotifier() : super(const AuthState()) {
    _loadStoredSession();
  }

  final _storage = const FlutterSecureStorage();
  final _apiClient = ApiClient();

  /// Check for a persisted token at startup.
  Future<void> _loadStoredSession() async {
    state = state.copyWith(isLoading: true);
    try {
      final token = await _storage.read(key: 'jwt_token');
      final name = await _storage.read(key: 'operator_name');
      if (token != null && token.isNotEmpty) {
        state = AuthState(
          isAuthenticated: true,
          token: token,
          operatorName: name,
        );
      } else {
        state = const AuthState();
      }
    } catch (_) {
      state = const AuthState();
    }
  }

  /// Login with operator credentials.
  Future<void> login(String username, String password) async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final response = await _apiClient.dio.post('/auth/login', data: {
        'username': username,
        'password': password,
      });

      final token = response.data['access_token'] as String;
      final name = response.data['operator_name'] as String? ?? username;

      await _apiClient.setToken(token);
      await _storage.write(key: 'operator_name', value: name);

      state = AuthState(
        isAuthenticated: true,
        token: token,
        operatorName: name,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: 'Login failed. Please check your credentials.',
      );
    }
  }

  /// Logout and clear stored credentials.
  Future<void> logout() async {
    await _apiClient.clearToken();
    await _storage.delete(key: 'operator_name');
    state = const AuthState();
  }
}

/// Global auth state provider.
final authProvider = StateNotifierProvider<AuthNotifier, AuthState>(
  (ref) => AuthNotifier(),
);

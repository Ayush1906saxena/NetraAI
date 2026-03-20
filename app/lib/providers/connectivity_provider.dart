import 'dart:async';

import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Simplified connectivity state.
enum NetworkStatus { online, offline }

/// Watches the device's network connectivity in real time.
class ConnectivityNotifier extends StateNotifier<NetworkStatus> {
  ConnectivityNotifier() : super(NetworkStatus.online) {
    _init();
  }

  late final StreamSubscription<List<ConnectivityResult>> _subscription;

  void _init() {
    // Check current status immediately.
    Connectivity().checkConnectivity().then(_update);

    // Subscribe to changes.
    _subscription = Connectivity().onConnectivityChanged.listen(_update);
  }

  void _update(List<ConnectivityResult> results) {
    if (results.contains(ConnectivityResult.none) || results.isEmpty) {
      state = NetworkStatus.offline;
    } else {
      state = NetworkStatus.online;
    }
  }

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }
}

/// Global connectivity provider.
final connectivityProvider =
    StateNotifierProvider<ConnectivityNotifier, NetworkStatus>(
  (ref) => ConnectivityNotifier(),
);

/// Convenience: true when the device has network access.
final isOnlineProvider = Provider<bool>(
  (ref) => ref.watch(connectivityProvider) == NetworkStatus.online,
);

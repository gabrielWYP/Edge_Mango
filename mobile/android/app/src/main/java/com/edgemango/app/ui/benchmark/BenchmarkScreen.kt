package com.edgemango.app.ui.benchmark

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel

/** Container: owns the view model and the Android-only pieces (file picker, share sheet). */
@Composable
fun BenchmarkScreen(modifier: Modifier = Modifier) {
    val vm: BenchmarkViewModel = viewModel()
    val state by vm.state.collectAsState()
    val context = LocalContext.current

    LaunchedEffect(vm) { vm.refresh() }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) vm.importModels(uris)
    }

    BenchmarkContent(
        state = state,
        actions = BenchActions(
            onSelectModel = vm::selectModel,
            onSelectAccelerator = vm::selectAccelerator,
            onWarmUp = vm::setWarmUp,
            onRuns = vm::setRuns,
            onUseLastFrame = vm::setUseLastFrame,
            onPickModel = { picker.launch(arrayOf("*/*")) },
            onRun = vm::run,
            onCancel = vm::cancel,
            onShare = { vm.shareIntent()?.let(context::startActivity) },
        ),
        modifier = modifier,
    )
}

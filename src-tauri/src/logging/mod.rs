//! Logging and error handling setup

use log::{error, info};
use std::panic;

/// Sets up a panic handler that logs the error before crashing
pub fn setup_panic_handler() {
    panic::set_hook(Box::new(|panic_info| {
        let location = panic_info.location().unwrap();
        let message = match panic_info.payload().downcast_ref::<&str>() {
            Some(s) => *s,
            None => match panic_info.payload().downcast_ref::<String>() {
                Some(s) => &s[..],
                None => "Box<dyn Any>",
            },
        };

        let err_msg = format!("Application crashed! Location: {}:{}, Message: {}", location.file(), location.line(), message);
        error!("{}", err_msg);
        
        // Write directly and synchronously to crash.log so it survives even if
        // the logger plugin's asynchronous file buffer has not flushed.
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            let crash_dir = std::path::PathBuf::from(local_app_data).join("com.arjun.workbench");
            let _ = std::fs::create_dir_all(&crash_dir);
            let _ = std::fs::write(crash_dir.join("crash.log"), &err_msg);
        }
    }));
    
    info!("Panic handler registered");
}

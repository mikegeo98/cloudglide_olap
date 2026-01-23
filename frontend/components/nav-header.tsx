"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Home, BarChart3, Settings } from "lucide-react"

export default function NavHeader() {
    const pathname = usePathname()

    const navItems = [
        {
            href: "/",
            label: "Simulation Wizard",
            icon: <Home className="h-4 w-4" />
        },
        {
            href: "/output",
            label: "Results Dashboard",
            icon: <BarChart3 className="h-4 w-4" />
        },
        {
            href: "/cache-explorer",
            label: "Cache Explorer",
            icon: <Settings className="h-4 w-4" />
        }
    ]

    return (
        <header className="w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center">
                <div className="mr-4 flex">
                    <Link href="/" className="mr-6 flex items-center space-x-2">
                        <span className="font-bold text-xl">CloudGlide</span>
                    </Link>
                </div>
                <nav className="flex items-center space-x-2 text-sm font-medium">
                    {navItems.map((item) => (
                        <Link key={item.href} href={item.href}>
                            <Button
                                variant={pathname === item.href ? "default" : "ghost"}
                                size="sm"
                                className="gap-2"
                            >
                                {item.icon}
                                {item.label}
                            </Button>
                        </Link>
                    ))}
                </nav>
            </div>
        </header>
    )
}
